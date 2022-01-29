<h1 align="center"> Super Terrain + </h1>
<p align="center"> A modern procedural terrain engine aiming for real-time and photorealism </p>

<p align="center">
	<img src="https://img.shields.io/badge/C%2B%2B_17-00599C?style=flat&logo=c%2B%2B&logoColor=white" />
	<img src="https://img.shields.io/badge/NVIDIA-CUDA_11-76B900?style=flat&logo=nvidia&logoColor=white" />
	<img src="https://img.shields.io/badge/OpenGL_4.6-FFFFFF?style=flat&logo=opengl" />
	</br>
	<img src="https://img.shields.io/badge/SQLite_3-07405E?style=flat&logo=sqlite&logoColor=white" />
	<img src="https://img.shields.io/badge/CMake_3-064F8C?style=flat&logo=cmake&logoColor=white" />
</p>

## :eyes: Overview

*SuperTerrain+* is a modern procedural terrain generator with physics simulations and aiming for real-time photorealistic rendering. It was started as my computer science dissertation project and inspired by many games, software and movies with procedural generation features. The time allowance as a school project is too limited to explore all the amazing stuff in this topic, therefore I would like to keep going.

Procedural generation is one of the most popular topics in computer graphics and allows us to create data using the power of algorithm and minimise efforts spent on pre-modelling/computing.

*SuperTerrain+* provides engine-like object-oriented interfaces to allow developers to customise the scenery as they preferred. Rather than considering it as a piece of software, it is also a collection of modern computer graphics techniques; with the help of detailed inline documentations, this is also a great place for learning.

*SuperTerrain+* is focusing on the next-generation computer graphics, so I have to spend most of my time on the latest technologies, therefore there is no intention for backward compatibility and the development environment such as programming language standard and hardware requirement, will be evolving over time.

## :bulb: Main Features

*SuperTerrain+* is definitely not the first application that brings out procedural generation, then why we need yet another one?

There is no perfect answer to this question, every application has its own pros and cons. Other than being modern, flexible and specialised for terrain generation and real-time photorealistic rendering, and of course for fun, these features are believed to differentiate *SuperTerrain+* from the others.

### Procedural heightfield infinite terrain

- [x] Tile-based infinite terrain
- [x] Improved simplex noise
- [x] Hardware instancing and tessellation
- [x] Continuous level-of-detail
- [x] Concurrent heightfield generation
- [x] Particle-based free-slip hydraulic erosion
- [x] Programmable static/runtime-compiled pipeline
- [x] Biome generation with classic *Minecraft* [Grown Biomes](http://cuberite.xoft.cz/docs/Generator.html) algorithm
- [x] Single histogram filter for Multi-biome heightfield generation with smooth transition
- [x] Rule-based biome-dependent terrain texture splatting with smooth transition
- [ ] River and lake generation

### Procedural volumetric infinite terrain

- [ ] Come up with a nice plan for this challenging topic

### Procedural geometry generation

- [ ] Rule-based geometry placement
- [ ] Procedural animated grassland generation
- [ ] Procedural parameter-based tree generation
- [ ] Volumetric cloud generation

### Real-time photorealistic rendering

- [x] Procedural atmospheric scattering
- [x] Realistic sun orbiting
- [ ] Post-processing
  - [x] Filmic tone mapping
  - [x] Gamma correction
  - [ ] Auto-exposure
  - [ ] Contrast
- [ ] Shadow mapping
  - [ ] Simple shadow mapping for spotlight
  - [ ] Cubemap shadow mapping for pointlight
  - [x] Parallel-split cascaded shadow mapping for directional light
- [ ] Deferred shading
- [ ] Screen-space ambient occlusion
- [ ] Anti-aliasing for deferred renderer
- [ ] Multiple lights
- [ ] Night rendering
- [ ] Procedural weather effect
- [ ] Screen-space reflection/refraction for water rendering
- [ ] ~~Real-time~~ raster-ray tracing hybrid rendering

### Optimisation technique

- [ ] Frustum culling
- [ ] Sparse virtual texture (a.k.a. mega-texture) for shadow map
- [ ] Deferred clustered lighting

## :bricks: Middleware

Those are some third-party libraries used by the main engine.

- [GLM](https://github.com/g-truc/glm)
- [GLAD](https://github.com/Dav1dde/glad)
- [SQLite3](https://www.sqlite.org/index.html)

Here is a list of additional dependencies if you are running the demo program.

- [GLFW](https://github.com/glfw/glfw)
- [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)
- [SIMPLE](https://github.com/stephen-hqxu/SIMPLE)

The application is unit-tested with.

- [Catch2 v3.0.0-preview4](https://github.com/catchorg/Catch2)

## :building_construction: Project Structure

### :card_index_dividers: Include directories

- CoreInterface
  - SuperTerrain+
- ModuleInterface
  - SuperAlgorithm+
  - SuperRealism+

### :dart: Build targets

- GLAD: Pre-compilation of `glad.c` for source and GL-context sharing among build targets.
- SuperTerrain+: Main procedural terrain generation engine.
  - SuperAlgorithm+: A library of useful algorithms for terrain generation.
    - SuperAlgorithm+Host: Algorithms that are best suited for CPU execution.
    - SuperAlgorithm+Device:  Algorithms that can be benefited from mass parallel computing using GPU.
  - SuperRealism+: A rendering engine for photorealistic rendering, with some handy GL functions and object wrappers.
- SuperDemo+: An application which demonstrates the usage of the main engine.
- SuperTest+: Unit test program.

### :ballot_box_with_check: CMake options

| Option | Explanation | Default |
| ------ | ------------ | ------- |
| STP_CUDA_RUNTIME_LIBRARY | Set the global `nvcc` compiler flag `-cudart=` to the value set | "Static" |
| STP_CUDA_VERBOSE_PTX | Append to the global `nvcc` compiler flag with `--ptxas-options=-v` | OFF |
| STP_USE_AVX2 | Use AVX2 instruction sets on all vector operations | ON |
| STP_BUILD_DEMO | Enable SuperDemo+ program build | ON |
| STP_BUILD_TEST | Enable SuperTest+ program build | OFF |

## :bookmark_tabs: Getting Started

### :desktop_computer: Prerequisites

**Hardware requirement**

- CPU
<p align="left">
	<img src="https://img.shields.io/badge/Intel-Core_i5_8400-0071C5?style=flat&logo=intel&logoColor=white" />
	<img src="https://img.shields.io/badge/AMD-Ryzen_5_2400G-ED1C24?style=flat&logo=amd&logoColor=white" />
</p>

- GPU
<p align="left">
	<img src="https://img.shields.io/badge/NVIDIA-GTX_1660-76B900?style=flat&logo=nvidia&logoColor=white" />
</p>

> Sorry for AMD and Intel GPU because they are not supported by CUDA. *SuperTerrain+* is mostly optimised for *Turing* (CUDA compute capability 7.5) and newer, consumer-level Nvidia GPUs. The GPU requirement here is a very trivial approximation, the performance depends highly on configuration and level of re-programming to the engine.

- RAM: 4GB
- OS
<p align="left">
	<img src="https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black" />
	<img src="https://img.shields.io/badge/Windows_10-0078D6?style=flat&logo=windows&logoColor=white" />
</p>

**OpenGL extension requirement**

- OpenGL 4.6 core profile
- GL_ARB_bindless_texture
- GL_ARB_shading_language_include
- GL_NV_gpu_shader5

> This is usually not a problem if your GPU meets the minimum system requirement and have a relatively recent driver installed.

### :gear: How to build

1. Master branch is considered as a *work-in-progress* branch in this project, so make sure you have grabbed the latest stable source code from [Releases](https://github.com/stephen-hqxu/superterrainplus/releases) :point_left: :grinning:

2. Unzip the source code and go to project root.

```sh

# x.x.x is the version number
unzip `superterrainplus-x.x.x.zip`
cd ./superterrainplus-x.x.x

```

3. Create and go to `build` folder.

```sh

mkdir build
cd ./build

```

4. Acquire project build script from CMake.

```sh

cmake ../

```

5. Configure *CMakeCache.txt* if preferred. Leave it as default otherwise.

6. Build the program

```sh

cmake --build ./ --config Release --target ALL_BUILD

```

You may obtain the following executables:

- `SuperDemo+` if demo build is enabled
- `SuperTest+` if test build is enabled

## :world_map: Project Management

[Project Roadmap](https://github.com/stephen-hqxu/superterrainplus/projects)

[Release Note](https://github.com/stephen-hqxu/superterrainplus/releases)

[Dissertation](https://github.com/stephen-hqxu/superterrainplus/tree/master/Report)

## :books: Credits

This section contains source code and libraries that are not dependencies of this project but they are where ideas and inspirations are taken from.

For academic style *BibTex* references, check out the project dissertation as linked above.

Please contact the project maintainer *Stephen Xu*(stephen.hqxu@gmail.com) if you find your publication is being used but not listed below.

### Terrain generation

- [Particle based hydraulic erosion](https://github.com/SebLague/Hydraulic-Erosion/tree/Coding-Adventure-E01)
- [Minecraft biome generator](https://github.com/KaptainWutax/BiomeUtils)
- [Linear time Gaussian filter](http://blog.ivank.net/fastest-gaussian-blur.html) by *Ivan Kutskir*
- [Stratified sampling technique](https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using) from *GPU Gems 2*

### Geometry generation

- [Animated grassland generation](https://github.com/spacejack/terra)

### Photorealistic rendering

- [High-level OpenGL function wrapper](https://github.com/cginternals/globjects)
- [Simulating the Colors of the Sky](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky) from *Scratchapixel*
- [Physically-based atmospheric scattering](https://github.com/wwwtyro/glsl-atmosphere/)
- [Cascaded Shadow Mapping](https://learnopengl.com/Guest-Articles/2021/CSM) by *M�rton �rb�cz* from *Learn OpenGL*
- [A Primer On Efficient Rendering Algorithms & Clustered Shading](http://www.aortiz.me/2018/12/21/CG.html#part-2) by *�ngel Ortiz*
- [Forward vs Deferred vs Forward+ Rendering with DirectX 11](https://www.3dgep.com/forward-plus/#Experiment_Setup_and_Performance_Results) by *Jeremiah van Oosten*

**Tone mapping**

- [Filmic tone mapping functions](https://bruop.github.io/tonemapping/) by *Bruno Opsenica*
- [HDR Theory and practice](https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp) by *Hajime Uchimura*
- [Advanced Techniques and Optimization of HDR Color Pipelines](http://32ipi028l5q82yhj72224m8j.wpengine.netdna-cdn.com/wp-content/uploads/2016/03/GdcVdrLottes.pdf) by *Timothy Lottes*
- [Uncharted2: HDR Lighting](http://slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting) by *John Hable*
- [Tone mapping curve sketch](https://www.shadertoy.com/view/WdjSW3)
- [Tone mapping rendering comparison](https://www.shadertoy.com/view/lslGzl)