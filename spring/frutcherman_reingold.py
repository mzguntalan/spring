from typing import Callable
from typing import NewType
from typing import Tuple

from jax import jit
from jax import numpy as jnp
from jax import random
from jaxtyping import Array

Vertices = NewType("Vertices", Array)  # type: ignore
Edges = NewType("Edges", Array)  # type:ignore
Displacements = NewType("Displacements", Array)  # type: ignore


@jit
def caclulate_displacements(
    vertices: Vertices, edges: Edges, k: float, temperature: Array
) -> Displacements:
    deltas = vertices[None, :, :] - vertices[:, None, :]
    norm_deltas = jnp.linalg.norm(deltas, axis=-1, keepdims=True)
    E = edges[:, :, None]

    attractive = E * deltas * jnp.pow(norm_deltas, 3)
    repulsive = -deltas * jnp.pow(k, 3)
    numerator = attractive + repulsive
    denominator = k * jnp.square(norm_deltas)

    individual_displacements = numerator / (denominator + 1e-17)

    displacements: Displacements = jnp.sum(individual_displacements, axis=1)

    unit_displacements: Displacements = displacements / (
        jnp.linalg.norm(displacements, axis=-1, keepdims=True) + 1e-17
    )

    scaling = jnp.clip(jnp.abs(displacements), None, temperature)

    final_displacements: Displacements = scaling * unit_displacements

    return final_displacements


def step(
    vertices: Vertices,
    edges: Edges,
    k: float,
    temperature: float,
    width: float,
    height: float,
) -> Tuple[Vertices, Displacements]:
    displacements = caclulate_displacements(vertices, edges, k, temperature)
    vertices += displacements

    x_lim = width
    y_lim = height
    lower_limits = jnp.array([-x_lim, -y_lim])
    higher_limits = jnp.array([x_lim, y_lim])
    vertices = jnp.clip(vertices, lower_limits, higher_limits)

    return vertices, displacements


def apply_frutcherman_reingold(
    key: random.PRNGKey,
    vertices: Vertices,
    edges: Edges,
    width: float,
    height: float,
    num_steps: int,
    cooling_fn: Callable[float, float],
    callback: Callable[[Vertices, Edges, int], None],
) -> Vertices:
    area = width * height
    k = jnp.sqrt(area / vertices.shape[0])

    for i in range(num_steps):
        t = i / num_steps
        temperature = cooling_fn(t)

        jitter_key, key = random.split(key)
        random_movement = (
            random.normal(jitter_key, vertices.shape) * jnp.sqrt(temperature) * 1e-2
        )

        vertices += random_movement
        vertices, displacements = step(vertices, edges, k, temperature, width, height)

        callback(vertices, edges, i)

    return vertices
