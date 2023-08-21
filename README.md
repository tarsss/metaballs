# Description

Simple metaballs meshing algorithm for realtime applications.

Can run in parallel using Unity's job system. Compatible with Burst compiler.

Surface search and voxelization works by seaching for one point on the blob's surface and flood filling from there. Iso function sampling accelerated by BVH tree. No spatial constraints.

Triangulation is based on Marching Cubes algorithm with slight modifications to create less degenerate triangles.

https://github.com/tarsss/metaballs/assets/65231359/10def490-5721-4561-8c9c-f1010361cb15

# TODO

- [ ] Try Naive Surface Nets for triangulation, should generate better geometry at lower resolutions
- [ ] Improve sampling vectorization: change data layout of node contents (Store 4-wide metaball structs), get rig of branching
- [ ] Find a way to reduce HashMap lookups count when propagating through surface
- [ ] Use SAH for more optimal nodes splitting

# References

https://paulbourke.net/geometry/polygonise/
