from jax import numpy as jnp

""" All Cooling functions admit a `t` in [0,1] """


def linear(t: float, start_temperature: float, end_temperature: float) -> float:
    return jnp.array((1 - t) * start_temperature + t * end_temperature)


def parabolic(t: float, max_temperature: float) -> float:
    a = 4 * max_temperature
    y = -a * jnp.square(t - 0.5) + max_temperature
    temperature: float = jnp.clip(y, 1e-17, max_temperature)
    return jnp.array(temperature)


def exponential_decay(
    t: float, max_temperature: float, min_temperature: float, k: float
) -> float:
    temperature: float = min_temperature + (
        max_temperature - min_temperature
    ) * jnp.exp(-k * t)
    return jnp.array(temperature)
