#  Release 0.1.0 - Initial 2D Generator

## Release highlight - 3D mesh terrain using 2D heightmap approach

- Fully adjustable terrain generation, highly parametrised.
- Physics-based hydraulic erosion using particles.
- Improved (by myself) simplex noise algorithm with more customisable gradients and random number generator.
- Terrain generated using CUDA, make sure you have a CUDA compatible GPU, with compute capability > 7.5.
- OpenGL 4.5 features are used, make sure your GPU can handle v4.5.

> You will need the OpenGL library to run the program, see glad.c attached.

- Multi-threading optimised.
- Infinite terrain with tile-based mesh (instead of regular chunk-based), with massively reduced bandwidth overhead.

## Future Plan

- Add CMake support so it can be compiled on different platform and doesn't need to rely on visual studio.
- Multi-biome generation with Minecraft biome generation algorithm (so-called "scaling algorithm"), with different parameters in different biomes. Currently in progress.
- Volumetric terrain generation using marching-cubes.
- Procedural planet generation.
- Ray-traced shadow and reflection with optix7.
- Volumetric cloud with ray-marching algorithm and Voronoi noise.

> All source codes are clearly documented.