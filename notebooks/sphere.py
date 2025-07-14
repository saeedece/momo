import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")

with app.setup:
    import gmsh
    import jax
    import jax.numpy as jnp
    import pyvista as pv

    from momo.geometry import Geometry
    from momo.operator import discretize_galerkin
    from momo.integrate import integrate


@app.cell
def init_geometry():
    lc = 1e-1

    # intialize gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()

    # add sphere
    sphere = gmsh.model.occ.addSphere(0, 0, 0, 0.5)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [sphere], name="sphere")

    # generate mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(), lc)
    gmsh.model.mesh.generate(2)

    # extract geometry object
    geo = Geometry.from_gmsh()

    # close gmsh
    gmsh.finalize()
    return (geo,)


@app.cell
def _(geo):
    a = discretize_galerkin(geo) / 8.854_187_818_8e-12
    b = jax.vmap(integrate, in_axes=(0, None))(geo.nodes[geo.triangles], lambda r: 2.0)
    rho = jnp.linalg.solve(a, b)
    cap = jnp.sum(rho * geo.areas / 2)
    cap
    return


if __name__ == "__main__":
    app.run()
