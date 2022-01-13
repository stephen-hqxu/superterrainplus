<h1 align="center"> Super Terrain + </h1>
<p align="center"> A modern open source real-time realistic terrain engine </p>

<p align="center">
	<img src="https://img.shields.io/badge/C%2B%2B_17-00599C?style=flat&logo=c%2B%2B&logoColor=white" />
	<img src="https://img.shields.io/badge/NVIDIA-CUDA_11-76B900?style=flat&logo=nvidia&logoColor=white" />
	<img src="https://img.shields.io/badge/SQLite_3-07405E?style=flat&logo=sqlite&logoColor=white" />
	<img src="https://img.shields.io/badge/OpenGL_4.6-FFFFFF?style=flat&logo=opengl" />
	</br>
	<img src="https://img.shields.io/badge/CMake_3-064F8C?style=flat&logo=cmake&logoColor=white" />
	<img src="https://img.shields.io/badge/Visual_Studio_2019-5C2D91?style=flat&logo=visual%20studio&logoColor=white" />
</p>

## :eyes: Overview

SuperTerrain+ is a terrain engine that incorporates procedural generation, physics simulation and photo-realistic rendering. It was originally my computer science undergraduate project and inspired by many games and movies with procedural generation features.

**Design Lemma**

- Realistic
- Procedural
- Pseudorandom generation
- Real-time rendering

## :sparkler: Main Features

### Procedural heightfield-based infinite terrain

- [x] Tile-based infinite chunk
- [x] Improved simplex noise
- [x] Hardware instancing and tessellation
- [x] Continuous level-of-detail
- [x] Concurrent rendering and generation
- [x] Particle-based free-slip hydraulic erosion
- [x] ~~Selective edge copy from rendering buffer~~
- [x] Programmable generator pipeline stage
- [x] Static-compiled pipeline stage
- [x] Runtime-compiled pipeline stage
- [x] Biome generation with classic Minecraft scaling algorithm
- [x] Multi-biome heightfield generation with smooth transition
- [x] Rule-based biome-dependent terrain texture splatting with smooth transition
- [ ] River and lake generation

### Procedural geometry generation

- [ ] Rule-based geometry placement
- [ ] Animated geometric grassland rendering
- [ ] Procedural parameter-based tree generation
- [ ] Volumetric cloud rendering

### Real-time realistic rendering

- [x] Procedural sun rendering
- [x] Procedural sky atmospheric effect rendering
- [x] Day-night cycle
- [x] High dynamic range with filmic tone mapping
- [ ] Soft and smooth shadow
- [ ] Night scene rendering
- [ ] Weather and seasonal effect
- [ ] Water reflection and refraction
- [ ] ~~Real-time raster-ray tracing hybrid rendering~~

## :bricks: Middleware

### Internal libraries

Those libraries are maintained by us and can be downloaded from our public repo :+1:.

- [SIMPLE](https://github.com/stephen-hqxu/SIMPLE)

### External libraries

Those are some third-party libraries used by this project, we always make sure the latest version is compatible with our project.

- [GLM](https://github.com/g-truc/glm)
- [GLFW](https://github.com/glfw/glfw)
- [GLAD](https://github.com/Dav1dde/glad)
- [stb](https://github.com/nothings/stb)
- [Catch2 v3.0.0-preview4](https://github.com/catchorg/Catch2)
- [SQLite3](https://www.sqlite.org/index.html)

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
  - SuperAlgorithm+: A library of useful noise algorithms for pipeline programming.
    - SuperAlgorithm+Host: Algorithms and prep-works for device computation that are best suited for CPU execution.
    - SuperAlgorithm+Device:  Algorithms that can be benefited from parallel compute.
  - SuperRealism+: A rendering engine for photo-realistic rendering, with some handy GL functions and object wrappers.
- SuperDemo+: An application which demonstrates the usage of the main engine.
- SuperTest+: Unit test program for dynamic testing.

### :ballot_box_with_check: CMake options

| Option | Explanation | Default |
| ------ | ------------ | ------- |
| STP_CUDA_RUNTIME_LIBRARY | Set the global `nvcc` compiler flag `-cudart=` to the value set | "Static" |
| STP_CUDA_VERBOSE_PTX | Append to the global `nvcc` compiler flag with `--ptxas-options=-v` | OFF |
| STP_USE_AVX2 | Use AVX2 instruction sets on all vector operations | ON |
| STP_BUILD_DEMO | Enable build SuperDemo+ program | ON |
| STP_BUILD_TEST | Enable build SuperTest+ program | OFF |

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

- RAM: 4GB
- OS
<p align="left">
	<img src="https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black" />
	<img src="https://img.shields.io/badge/Windows_10-0078D6?style=flat&logo=windows&logoColor=white" />
</p>

**OpenGL extensions**

Your GPU must support the following OpenGL extensions:

- OpenGL 4.6 core profile
- GL_ARB_bindless_texture
- GL_ARB_shading_language_include

### :gear: How to build

1. Master branch is not guaranteed to be stable, so make sure you have grabbed the latest stable source code from [**Releases**](https://github.com/stephen-hqxu/superterrainplus/releases) :point_left: :grinning:

2. Unzip the source code and go to project root.

```sh

unzip `superterrainplus-master.zip`
cd ./superterrainplus-master

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

5. Configure CMake cache if preferred. Leave it as default otherwise.

6. Build the program

```sh

cmake --build ./ --config Release --target ALL_BUILD

```

Executables:

- `SuperDemo+` if demo build is enabled
- `SuperTest+` if test build is enabled

## :world_map: Project Management

[Project Roadmap](https://github.com/stephen-hqxu/superterrainplus/projects)

[Release Note](https://github.com/stephen-hqxu/superterrainplus/releases)

[Dissertation](https://github.com/stephen-hqxu/superterrainplus/tree/master/Report)

## :seedling: Contribution

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

## :telephone: Contact

Stephen Xu - stephen.hqxu@gmail.com

Project Repository: https://github.com/stephen-hqxu/superterrainplus

## :books: Credit

Reference contains source code and libraries that are not dependencies of this project but they are where ideas, inspirations and design patterns taken from.

- [Minecraft biome generator](https://github.com/KaptainWutax/BiomeUtils)
- [Particle based hydraulic erosion](https://github.com/SebLague/Hydraulic-Erosion/tree/Coding-Adventure-E01)
- [High-level OpenGL function wrapper](https://github.com/cginternals/globjects)
- [Animated grass geometries rendering](https://github.com/spacejack/terra)
- [Physically-based atmospheric scattering](https://github.com/wwwtyro/glsl-atmosphere/)
- [Integer-format texture smoothing](https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using)
- [The sun, sky, cloud and night rendering](https://www.shadertoy.com/view/tdSXzD)
- [Filmic tone mapping functions](https://bruop.github.io/tonemapping/)

For a more academic-styled reference, check the project dissertation as linked above.