"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=bare-except, consider-using-generator, ungrouped-imports
"""Utils that are only interesting to MaxText. """
from typing import NamedTuple, Any

import jax
import re
import chex

import optax
from optax._src import utils as optax_utils
from optax._src import linear_algebra as optax_linear_algebra
import jax.numpy as jnp
import numpy as np

from max_utils import create_learning_rate_schedule

def tree_path_to_string(path, sep=None):
  keys = []
  for key in path:
    if isinstance(key, jax.tree_util.SequenceKey):
      keys.append(str(key.idx))
    elif isinstance(key, jax.tree_util.DictKey):
      keys.append(str(key.key))
    elif isinstance(key, jax.tree_util.GetAttrKey):
      keys.append(str(key.name))
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
      keys.append(str(key.key))
    else:
      keys.append(str(key))
  if sep is None:
    return tuple(keys)
  return sep.join(keys)

def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
  """ An extended version of jax.tree_util.tree_map, where the mapped function
      f takes both the name (path) and the tree leaf as input.
  """
  return jax.tree_util.tree_map_with_path(
    lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
    tree, *rest,
    is_leaf=is_leaf
  )

def get_weight_decay_mask(exclusions):
  """ Return a weight decay mask function that computes the pytree masks
      according to the given exclusion rules.
  """
  def decay(name, p):
    for rule in exclusions:
      if re.search(rule, name) is not None:
        print((name, p.shape, p.dtype, "not use weight_decay"))
        return False
      
    print((name, p.shape, p.dtype, "use weight_decay"))
    return True

  def weight_decay_mask(params):
    return named_tree_map(decay, params, sep='/')
  
  return weight_decay_mask

def abs_sq(x):
  """Returns the squared norm of a (maybe complex) array.

  For real `x`, JAX generates the same HLO from this, `jnp.square(x)`, `x * x`,
  or `x**2`.

  Args:
    x: a (maybe complex) array.

  Returns:
    The squared norm of `x`.
  """
  if not isinstance(x, (np.ndarray, jnp.ndarray)):
    raise ValueError(f"`abs_sq` accepts only NDarrays, got: {x}.")
  return (x.conj() * x).real

def safe_root_mean_squares(x, min_rms, axis=None, keepdims=False):
  """Returns `maximum(sqrt(mean(abs_sq(x))), min_norm)` with correct grads.

  The gradients of `maximum(sqrt(mean(abs_sq(x))), min_norm)` at 0.0
  is `NaN`, because jax will evaluate both branches of the `jnp.maximum`. This
  function will instead return the correct gradient of 0.0 also in such setting.

  Args:
    x: jax array.
    min_rms: lower bound for the returned norm.

  Returns:
    The safe RMS of the input vector, accounting for correct gradient.
  """
  rms = jnp.sqrt(jnp.mean(abs_sq(x), axis=axis, keepdims=keepdims))
  x = jnp.where(rms <= min_rms, jnp.ones_like(x), x)
  return jnp.where(rms <= min_rms, min_rms, jnp.sqrt(jnp.mean(abs_sq(x), axis=axis, keepdims=keepdims)))

def get_optimizer(config):
  """learning rate schedule"""
  learning_rate_schedule = create_learning_rate_schedule(
    config, 
    step_reduction=config.gradient_accumulation_steps,
  )

  if config.opt_type == "tiger":
    return tiger_adam_pax(
      learning_rate=learning_rate_schedule,
      tiger_beta=config.tiger_b,
      tiger_weight_decay=config.tiger_weight_decay,
      adam_learning_rate_fraction=config.adam_learning_rate_fraction,
      adam_b1=config.adam_b1,
      adam_b2=config.adam_b2,
      adam_eps=config.adam_eps,
      adam_eps_root=config.adam_eps_root,
      adam_weight_decay=config.adam_weight_decay,
      gradient_accumulation_steps=config.gradient_accumulation_steps,
      mu_dtype=jnp.float32,
    ), learning_rate_schedule

  if config.opt_type == "sgd":
    optimizer = [
      optax.identity(),
      optax.scale_by_learning_rate(learning_rate_schedule)
    ]
  elif config.opt_type == "adamw":
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    optimizer = [
      optax.scale_by_adam(
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        eps_root=config.adam_eps_root,
        mu_dtype=jnp.float32,
        nesterov=False,
      ),
      optax.add_decayed_weights(
        weight_decay=config.adam_weight_decay, 
        mask=get_weight_decay_mask([
          "norm",
          "scale",
          "bias",
        ])
      ),
      optax.scale_by_learning_rate(learning_rate_schedule),
    ]
  elif config.opt_type == "adam_pax":
    optimizer = [
      optax.identity(),
      adam_pax(
        learning_rate_schedule,
        beta1=config.adam_b1,
        beta2=config.adam_b2,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
      )
    ]
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")
  
  # gradient_clipping_threshold
  if config.gradient_clipping_threshold > 0:
    optimizer = [
      optax.clip_by_global_norm(config.gradient_clipping_threshold)
    ] + optimizer
  
  optimizer = optax.chain(*optimizer)
  
  # gradient_accumulation_steps
  if config.gradient_accumulation_steps > 1:
    optimizer = optax.MultiSteps(
        optimizer, config.gradient_accumulation_steps
    )

  return optimizer, learning_rate_schedule

def tiger_adam_pax(
  learning_rate: optax.Schedule,
  tiger_beta: float,
  tiger_weight_decay: float = 1e-3,
  adam_learning_rate_fraction: float = 10.0,
  adam_b1: float = 0.9,
  adam_b2: float = 0.95,
  adam_eps: float = 1.e-8,
  adam_eps_root: float = 0.,
  adam_weight_decay: float = 0.1,
  mu_dtype = None,
  gradient_accumulation_steps: int = 1,
):
  # tiger
  tiger_optimizer = tiger_pax(
    learning_rate=learning_rate,
    beta=tiger_beta,
    mu_dtype=mu_dtype,
    weight_decay=tiger_weight_decay,
    gradient_accumulation_steps=gradient_accumulation_steps,
  )

  # adam
  def adam_learning_rate(step):
    return learning_rate(step) * adam_learning_rate_fraction

  adam_optimizer = optax.chain(
    optax.scale_by_adam(
      b1=adam_b1,
      b2=adam_b2,
      eps=adam_eps,
      eps_root=adam_eps_root,
      mu_dtype=None,
      nesterov=False,
    ),
    optax.add_decayed_weights(
      weight_decay=adam_weight_decay, 
      mask=get_weight_decay_mask([
        "norm",
        "scale",
        "bias",
      ])
    ),
    optax.scale_by_trust_ratio(),
    optax.scale_by_learning_rate(adam_learning_rate),
  )

  if gradient_accumulation_steps > 1:
    adam_optimizer = optax.MultiSteps(
        adam_optimizer, gradient_accumulation_steps
    )

  # 融合召唤
  def is_emb(name):
    if (
      "embedding" in name
      or "logits_dense" in name
    ):
      return True
    
    return False

  def mixed_init_fn(params):
    emb_params = named_tree_map(lambda n, p: p if is_emb(n) else None, params, sep='/')
    trunk_params = named_tree_map(lambda n, p: None if is_emb(n) else p, params, sep='/')
    return (
      adam_optimizer.init(emb_params), 
      tiger_optimizer.init(trunk_params), 
    )

  def mixed_update_fn(
    updates, 
    state, 
    params,
  ):
    emb_updates = named_tree_map(lambda n, u: u if is_emb(n) else None, updates, sep='/')
    trunk_updates = named_tree_map(lambda n, u: None if is_emb(n) else u, updates, sep='/')

    emb_params = named_tree_map(lambda n, p: p if is_emb(n) else None, params, sep='/')
    trunk_params = named_tree_map(lambda n, p: None if is_emb(n) else p, params, sep='/')

    emb_state, trunk_state = state

    # use update
    emb_updates, emb_new_state = adam_optimizer.update(emb_updates, emb_state, emb_params)
    trunk_updates, trunk_new_state = tiger_optimizer.update(trunk_updates, trunk_state, trunk_params)

    # mixed update
    updates = named_tree_map(lambda n, u, eu, tu: eu if is_emb(n) else tu, updates, emb_updates, trunk_updates, sep='/')

    return updates, (
      emb_new_state, 
      trunk_new_state, 
    )

  return optax.GradientTransformation(mixed_init_fn, mixed_update_fn)


class TigerState(NamedTuple):
  """State for the Lion algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mini_step: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates

def tiger_pax(
  learning_rate: optax.Schedule,
  beta: float,
  mu_dtype = None,
  weight_decay: float = 1e-3,
  gradient_accumulation_steps: int = 1,
):
  mu_dtype = optax_utils.canonicalize_dtype(mu_dtype)

  def base_init_fn(params):
    mu = jax.tree_util.tree_map(  # moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    return TigerState(
      count=jnp.zeros([], jnp.int32), 
      mini_step=jnp.zeros([], dtype=jnp.int32),
      mu=mu
    )

  def base_update_fn(
    updates, 
    state: TigerState, 
    params=None,
  ):
    beta1 = jnp.where(state.mini_step == 0, beta, jnp.ones_like(beta))
    beta2 = (1 - beta) / gradient_accumulation_steps
    new_mu = jax.tree_util.tree_map(
      lambda g, t: beta1 * t + beta2 * g, 
      updates, state.mu
    )
    new_mu = optax_utils.cast_tree(new_mu, mu_dtype)

    def _do_update(mu, params, count):
      step_size = -learning_rate(count)

      def _name_update(name, m, p):
        # base update
        u = jnp.sign(m)
        # decayed_weights
        if (
          "norm" in name
          or "scale" in name
          or "bias" in name
        ):
          u = u
          print((name, p.shape, p.dtype, "not use weight decay in tiger"))
        else:
          u = u + weight_decay * p
          print((name, p.shape, p.dtype, "use weight decay in tiger"))

        # scale
        if (
          "norm" in name
          or "scale" in name
          or "bias" in name
        ):
          print((name, p.shape, p.dtype, "use 0.5 scale"))
          scale = 0.5
        elif (
          "layers" in name
        ):
          mean_axis = [d for d in range(p.ndim) if d != 1]
          print((name, p.shape, p.dtype, f"use layers root_mean_square at axis={mean_axis} scale"))

          param_norm = optax.safe_norm(p, 0.0, ord=2, axis=mean_axis, keepdims=True)
          update_norm = optax.safe_norm(u, 0.0, ord=2, axis=mean_axis, keepdims=True)
          trust_ratio = param_norm / update_norm

          scale = jnp.where(jnp.logical_or(param_norm == 0., update_norm == 0.), jnp.array(1.0, dtype=p.dtype), trust_ratio)
        else:
          print((name, p.shape, p.dtype, "use base root_mean_square scale"))

          param_norm = optax.safe_norm(p, 0.0, ord=2)
          update_norm = optax.safe_norm(u, 0.0, ord=2)
          trust_ratio = param_norm / update_norm

          scale = jnp.where(jnp.logical_or(param_norm == 0., update_norm == 0.), jnp.array(1.0, dtype=p.dtype), trust_ratio)
          
        return jnp.array(step_size, dtype=u.dtype) * scale * u

      return named_tree_map(_name_update, mu, params, sep='/'), optax.safe_int32_increment(count)

    def _skip_update(mu, params, count):
      del params
      return jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), mu), count
    
    updates_new, count_new = jax.lax.cond(
      state.mini_step == (gradient_accumulation_steps - 1), _do_update, _skip_update, *(new_mu, params, state.count)
    )
    mini_step_new = optax.safe_int32_increment(state.mini_step) % gradient_accumulation_steps

    return updates_new, TigerState(
      count=count_new, 
      mini_step=mini_step_new,
      mu=new_mu,
    )

  return optax.GradientTransformation(base_init_fn, base_update_fn)
  

def adam_pax(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    ) -> optax.GradientTransformation:
  """Standard Adam optimizer that supports weight decay.

  Follows the implemenation in pax/praxis sharded_adam
  https://github.com/google/praxis/blob/545e00ab126b823265d70c715950d39333484f38/praxis/optimizers.py#L621

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `optax.GradientTransformation`.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        jnp.zeros_like, params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step.astype(jnp.float32) + 1.
    return decay * (1. - jnp.power(decay, t - 1.)) / (1. - jnp.power(decay, t))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    count = state.count

    class _slot_opt_state:
      def __init__(self, mu, nu):
        self.mu = mu
        self.nu = nu

    def _update_momentum(update, mu, nu):
      # The conversion to the data type of the update ensures that bfloat16 remains
      # bfloat16 in the optimizer state. This conversion has to be done after
      # `bias_corrected_dacay` is calculated as calculating `jnp.power(decay, t)` in low
      # precision can result in it being rounded to 1 and subsequently a
      # "division by zero" error.
      beta1_decay = bias_corrected_decay(count, beta1).astype(update)
      beta2_decay = bias_corrected_decay(count, beta2).astype(update)
      mu = (1.0 - beta1_decay) * update + beta1_decay * mu
      nu = (1.0 - beta2_decay) * (update**2) + beta2_decay * nu
      return _slot_opt_state(mu=mu, nu=nu)

    updated_moments = jax.tree_map(_update_momentum, updates, state.mu, state.nu)

    mu = jax.tree_map(lambda x: x.mu, updated_moments)
    nu = jax.tree_map(lambda x: x.nu, updated_moments)

    updates = jax.tree_map(
        lambda mu, nu: mu / (jnp.sqrt(nu + epsilon_root) + epsilon), mu, nu)

    if weight_decay > 0:
      updates = jax.tree_map(lambda x, v: x + weight_decay * v, updates, params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
