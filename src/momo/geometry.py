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

from __future__ import annotations
from functools import cached_property

import gmsh
import jax
import jax.numpy as jnp
import pyvista as pv
from chex import dataclass
from jax import Array


@dataclass
class Geometry:
    nodes: Array
    edges: Array
    triangles: Array
    triangles_edges: Array

    @classmethod
    def from_gmsh(cls):
        _, node_coords, _ = gmsh.model.mesh.getNodes(dim=-1, tag=1)
        _, _, node_tags = gmsh.model.mesh.getElements(dim=2, tag=-1)
        physical_ids = gmsh.model.getPhysicalGroups(2)

        nodes = jnp.asarray(node_coords).reshape(-1, 3)
        triangles = jnp.asarray(node_tags, dtype=jnp.uint32).reshape(-1, 3) - 1

        @jax.jit
        def build_rwg_bases(triangles: Array) -> tuple[Array, Array]:
            num_triangles = triangles.shape[0]
            owners = jnp.repeat(jnp.arange(num_triangles, dtype=jnp.uint32), 3)
            node_pairs = (
                triangles[:, (0, 1, 0, 2, 1, 2)]
                .reshape(3 * num_triangles, 2)
                .astype(jnp.uint64)
            )
            node_pairs = jnp.sort(node_pairs, axis=1)
            key = (node_pairs[:, 0] << 32) | node_pairs[:, 1]
            order = jnp.argsort(key)

            node_owners = owners[order].reshape(-1, 2)
            node_values = (
                jnp.column_stack((key[order[::2]] >> 32, key[order[::2]] & 0xFFFFFFFF))
                .reshape(-1, 2)
                .astype(jnp.uint32)
            )

            updated_key = (node_values[:, 0].astype(jnp.uint64) << 32) | node_values[
                :, 1
            ].astype(jnp.uint64)

            edges = jnp.concat((node_values, node_owners), axis=1)
            triangles_edges = jnp.searchsorted(updated_key, key).reshape(-1, 3)
            return edges, triangles_edges

        edges, triangles_edges = build_rwg_bases(triangles)
        return cls(
            nodes=nodes,
            edges=edges,
            triangles=triangles,
            triangles_edges=triangles_edges,
        )

    def to_vtk(self):
        cell_types = jnp.full_like(
            self.triangles,
            fill_value=pv.CellType.TRIANGLE,
            dtype=jnp.uint8,
        )
        grid = pv.UnstructuredGrid(self.triangles, cell_types, self.nodes)
        return grid

    @cached_property
    def centroids(self) -> Array:
        return jnp.mean(self.nodes[self.triangles, :], axis=1)

    @cached_property
    def areas(self) -> Array:
        v1, v2 = self.nodes[self.triangles, 0], self.nodes[self.triangles, 1]
        areas = 0.5 * jnp.linalg.norm(jnp.cross(v1, v2), axis=1)
        return areas
