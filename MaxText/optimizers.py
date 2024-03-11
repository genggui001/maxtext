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
from typing import NamedTuple

import jax
import re
import chex

import optax
from optax._src import utils as optax_utils
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
  def decay(name, _):
    for rule in exclusions:
      if re.search(rule, name) is not None:
        print((name, "not use weight_decay"))
        return False
      
    print((name, "use weight_decay"))
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
    return tiger_pax(
      learning_rate=learning_rate_schedule,
      beta=config.adam_b1,
      weight_decay=config.adam_weight_decay,
      gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        mu_dtype=None,
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
  elif config.opt_type == "lion":
    optimizer = [
      optax.scale_by_lion(
        b1=config.adam_b1,
        b2=config.adam_b2,
        mu_dtype=None,
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

class TigerState(NamedTuple):
  """State for the Lion algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mini_step: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates

def tiger_pax(
  learning_rate: optax.Schedule,
  beta: float,
  powerball_gamma: float = 0.5,
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
        if (
          "embedding" in name
          or "logits_dense" in name
        ):
          u = jnp.sign(m) * (jnp.abs(m) ** powerball_gamma)
          print((name, "use powerball sign"))
        else:
          u = jnp.sign(m)
          print((name, "use sign"))

        # decayed_weights
        if (
          "norm" in name
          or "scale" in name
          or "bias" in name
        ):
          u = u
          print((name, "not use weight decay"))
        else:
          u = u + weight_decay * p
          print((name, "use weight decay"))
          
        # scale
        if (
          "norm" in name
          or "scale" in name
          or "bias" in name
        ):
          print((name, p.shape, "use 0.5 scale"))
          scale = 0.5
        elif (
          "layers" in name
        ):
          mean_axis = [d for d in range(p.ndim) if d != 1]
          print((name, p.shape, f"use layers root_mean_square at axis={mean_axis} scale"))
          p_norm = safe_root_mean_squares(p, min_rms=0., axis=mean_axis, keepdims=True)
          scale = jnp.where(p_norm == 0., jnp.array(1.0, dtype=p.dtype), p_norm)
        else:
          print((name, p.shape, "use base root_mean_square scale"))
          p_norm = safe_root_mean_squares(p, min_rms=0.)
          scale = jnp.where(p_norm == 0., jnp.array(1.0, dtype=p.dtype), p_norm)

        return jnp.array(step_size, dtype=u.dtype) * scale * u

      return named_tree_map(_name_update, mu, params, sep='/'), optax.safe_int32_increment(count)

    def _skip_update(mu, params, count):
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

  return optax.chain(
    optax.identity(),
    optax.GradientTransformation(base_init_fn, base_update_fn),
  )
  

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
      beta1_decay = bias_corrected_decay(count, beta1)
      beta2_decay = bias_corrected_decay(count, beta2)
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
