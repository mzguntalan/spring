from itertools import product

from jax import numpy as jnp
from jaxtyping import Array

from spring.frutcherman_reingold import Edges


def adjacency_matrix_to_list(edges: Edges) -> Array:
    edge_list = []
    for i, j in product(range(edges.shape[0]), range(edges.shape[0])):
        if j < i:
            continue
        if edges[i, j] != 0:
            edge_list.append([i, j])

    return edge_list
