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
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

_ALPHA = jnp.asarray(
    [
        0.0915762135098,
        0.8168475729805,
        0.0915762135098,
        0.1081030181681,
        0.4459484909160,
        0.4459484909160,
    ]
)

_BETA = jnp.asarray(
    [
        0.0915762135098,
        0.0915762135098,
        0.8168475729805,
        0.4459484909160,
        0.1081030181681,
        0.4459484909160,
    ]
)

_WEIGHTS_TRI = jnp.asarray(
    [
        0.0549758718276,
        0.0549758718276,
        0.0549758718276,
        0.1116907948390,
        0.1116907948390,
        0.1116907948390,
    ]
)

_WEIGHTS_LINE = jnp.asarray(
    [0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727]
)

_NODES_LINE = jnp.asarray(
    [0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797027]
)


@jax.jit
def green(r: Array, rp: Array) -> Array:
    rr = jnp.linalg.norm(r - rp)
    return 1 / (4 * jnp.pi * rr)


@jax.jit
def green_no_singularity(r: Array, rp: Array) -> Array:
    rr = jnp.linalg.norm(r - rp)
    return 1 / (4 * jnp.pi) * jnp.ones_like(rr)


@jax.jit
def area(triangle: Array) -> Array:
    return 0.5 * jnp.linalg.norm(
        jnp.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    )


@jax.jit
def quadrature(triangle: Array) -> Array:
    nodes = (
        (1 - _ALPHA - _BETA)[:, jnp.newaxis] * triangle[jnp.newaxis, 0]
        + _ALPHA[:, jnp.newaxis] * triangle[jnp.newaxis, 1]
        + _BETA[:, jnp.newaxis] * triangle[jnp.newaxis, 2]
    )
    return nodes


@partial(jax.jit, static_argnames=["integrand"])
def integrate(triangle: Array, integrand: Callable[..., Array]) -> Array:
    nodes = quadrature(triangle)
    integral = integrand(nodes) * _WEIGHTS_TRI
    integral = jnp.sum(integral) * 2 * area(triangle)
    return integral


@partial(jax.jit, static_argnames=["integrand"])
def integrate_double(
    triangle_tst: Array,
    triangle_src: Array,
    integrand: Callable[..., Array],
):
    nodes_tst = quadrature(triangle_tst)[:, jnp.newaxis, :]
    nodes_src = quadrature(triangle_src)[jnp.newaxis, :, :]
    integral = (
        nodes_tst
        * nodes_src
        * _WEIGHTS_TRI[:, jnp.newaxis, jnp.newaxis]
        * _WEIGHTS_TRI[jnp.newaxis, :, jnp.newaxis]
    ) * integrand(nodes_tst, nodes_src)
    integral = jnp.sum(integral) * 4 * area(triangle_tst) * area(triangle_src)
    return integral


@partial(jax.vmap, in_axes=(None, None, 0))
@partial(jax.jit, static_argnames=["integrand"])
def integrate_near_singularity(
    triangle: Array,
    integrand: Callable[..., Array],
    observation_point: Array,
):
    injected_integrand = lambda r: integrand(r, observation_point)
    normal = jnp.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= jnp.linalg.norm(triangle)

    z_dir = jnp.dot(observation_point - triangle[0], normal)
    z_unit = jax.lax.select(jnp.sign(z_dir) > 0, normal, -normal)
    projected_observation_point = observation_point - z_dir * normal

    triangle_area = area(triangle)
    barycentric_coords = jnp.sum(
        z_unit
        * jnp.cross(
            triangle[(1, 2, 0), :] - projected_observation_point[jnp.newaxis, :],
            triangle[(2, 0, 1), :] - projected_observation_point[jnp.newaxis, :],
        ),
        axis=1,
    ) / (2 * triangle_area)

    sub_triangles = jnp.concat(
        (
            jnp.repeat(projected_observation_point, 3).reshape(3, 3).T,
            triangle[(1, 2, 0), :],
            triangle[(2, 0, 1), :],
        ),
        axis=1,
    ).reshape(3, 3, 3)

    barycentric_sgn = jnp.sign(barycentric_coords) >= 0

    local_x_axes = jax.lax.select(
        jnp.column_stack((barycentric_sgn, barycentric_sgn, barycentric_sgn)),
        sub_triangles[:, 1] - sub_triangles[:, 2],
        sub_triangles[:, 2] - sub_triangles[:, 1],
    )
    local_x_axes /= jnp.linalg.norm(local_x_axes, axis=1)
    local_y_axes = jnp.cross(normal, local_x_axes)
    local_y_axes /= jnp.linalg.norm(local_y_axes, axis=1)

    side1 = sub_triangles[:, 1] - sub_triangles[:, 0]
    side2 = sub_triangles[:, 2] - sub_triangles[:, 0]

    phi1 = jnp.arccos(
        jnp.sum(local_x_axes * side1, axis=1)
        / (jnp.linalg.norm(local_x_axes, axis=1) * jnp.linalg.norm(side1, axis=1))
    )
    phi2 = jnp.arccos(
        jnp.sum(local_x_axes * side2, axis=1)
        / (jnp.linalg.norm(local_x_axes, axis=1) * jnp.linalg.norm(side2, axis=1))
    )
    phi_lower = jax.lax.select(barycentric_sgn, phi1, phi2)
    phi_upper = jax.lax.select(barycentric_sgn, phi2, phi1)
    h = jax.lax.select(
        barycentric_sgn,
        jnp.linalg.norm(side1, axis=1),
        jnp.linalg.norm(side2, axis=1),
    ) * jnp.sin(phi_lower)
    zeta_lower = jnp.log(jnp.tan(phi_lower / 2))
    zeta_upper = jnp.log(jnp.tan(phi_upper / 2))

    # broadcasting hell but no loops waow!
    zeta = (zeta_upper - zeta_lower)[:, jnp.newaxis] * _NODES_LINE[
        jnp.newaxis, :
    ] + zeta_lower[:, jnp.newaxis]
    phi = 2 * jnp.arctan(jnp.exp(zeta))

    r_upper = jnp.sqrt(jnp.square(z_dir) + jnp.square(h[:, jnp.newaxis] / jnp.sin(phi)))
    r_lower = jnp.abs(z_dir) * jnp.ones_like(r_upper)
    eta = _NODES_LINE[jnp.newaxis, jnp.newaxis, :]
    src_weight = (
        _WEIGHTS_LINE[jnp.newaxis, :, jnp.newaxis]
        * _WEIGHTS_LINE[jnp.newaxis, jnp.newaxis, :]
        * (zeta_upper - zeta_lower)[:, jnp.newaxis, jnp.newaxis]
    )
    rp_magnitude = (
        eta * (r_upper - r_lower)[..., jnp.newaxis] + r_lower[..., jnp.newaxis]
    )

    rp = jnp.sqrt(jnp.square(rp_magnitude) - (z_dir**2) * jnp.ones(shape=(3, 4, 4)))
    src_nodes = projected_observation_point[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, :
    ] + local_x_axes[:, jnp.newaxis, jnp.newaxis, :] * rp[..., jnp.newaxis] * jnp.cos(
        phi[..., jnp.newaxis, jnp.newaxis]
    )

    # flatten nodes out so we can use a single vmap
    src_nodes_shape = src_nodes.shape
    src_nodes = src_nodes.reshape(-1, 3)

    # integrate finally
    integral = (
        jax.vmap(injected_integrand, in_axes=0)(src_nodes).reshape(3, 4, 4)
        * ((r_upper - r_lower) / jnp.cosh(zeta))[..., jnp.newaxis]
        * src_weight
        * jnp.sign(barycentric_coords)[..., jnp.newaxis, jnp.newaxis]
    ).sum()

    return integral
