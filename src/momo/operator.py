"""
Momo: A tiny method of moments library.
Copyright (C) 2025 saeedece

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import Geometry
from .integrate import (
    green,
    green_no_singularity,
    integrate_double,
    integrate_near_singularity,
    quadrature,
    _WEIGHTS_TRI,
)


@jax.jit
def discretize_galerkin(geometry: Geometry):
    triangles = geometry.nodes[geometry.triangles]
    mask = jnp.linalg.norm(
        geometry.centroids[..., jnp.newaxis, jnp.newaxis]
        - geometry.centroids[jnp.newaxis, jnp.newaxis, ...],
        axis=(1, 3),
    )
    mask = mask < 2e-1

    nodes = jax.vmap(quadrature, in_axes=0)(triangles)
    weights = _WEIGHTS_TRI[jnp.newaxis, jnp.newaxis, :]
    integral_near_singularity = jnp.sum(
        (
            jax.vmap(
                jax.vmap(integrate_near_singularity, in_axes=(None, None, 0)),
                in_axes=(0, None, None),
            )(triangles, green_no_singularity, nodes)
            * 2
            * geometry.areas[:, jnp.newaxis]
        )
        * weights,
        axis=-1,
    )

    integral_far_singularity = jax.vmap(
        jax.vmap(integrate_double, in_axes=(None, 0, None)), in_axes=(0, None, None)
    )(triangles, triangles, green)

    return jax.lax.select(mask, integral_near_singularity, integral_far_singularity)
