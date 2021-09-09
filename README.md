<h1 align="center"> Super Terrain + </h1>
<p align="center"> An open source real-time realistic terrain engine </p>

## :eyes: Overview

Super Terrain + (or STP in short) is a procedural terrain generation engine that incorporates natural physics-simulation and photo-realistic rendering. It was originally my computer science undergraduate project and inspired by games with random generation features, most noticeably Minecraft.

**Design Lemma**

- Realistic
- Real-time capable
- Procedural
- Deterministically random

## :sparkler: Main Features

### Procedural 2D infinite terrain

- [x] Tile-based infinite chunk
- [x] Improved simplex noise algorithm
- [x] Hardware instancing and tessellation
- [x] Dynamic Level-of-Detail
- [x] Concurrent rendering and generation
- [x] Particle-based free-slip hydraulic erosion
- [x] Selective edge copy from rendering buffer
- [x] Programmable generator pipeline stage
- [x] Static-compiled pipeline stage
- [x] Runtime-compiled pipeline stage
- [x] Biome generation with classic Minecraft scaling algorithm
- [x] Different terrain shape based on generated biome
- [ ] Altitude, gradient and biome dependent texture splating
- [ ] Biome feature generation
- [ ] Real-time ray tracing rendering

### Procedural entity geneartion

- [ ] Volumetric cloud rendering
- [ ] Procedural foilage generation and placement

### Procedural 3D (volumetric) infinite terrain

No plan, yet.

## :bricks: Middleware

### Language

- C++ 17
- CUDA 11.3
- OpenGL 4.6
- CMake 3.18

### Internal libraries

Those libraries are maintained by us and can be downloaded from our public repo :+1:.

- [SIMPLE](https://github.com/stephen-hqxu/SIMPLE)
- [SglToolkit](https://github.com/stephen-hqxu/SglToolkit)

### External libraries

Those are some third-party libraries used by this project, we always make sure the latest version is compatible with our project.

- [GLM](https://github.com/g-truc/glm)
- [GLFW](https://github.com/glfw/glfw)
- [GLAD](https://github.com/Dav1dde/glad)
- [stb](https://github.com/nothings/stb)
- [Catch2 v3](https://github.com/catchorg/Catch2)

## :building_construction: Project Structure

### :card_index_dividers: Include directories

- CoreInterface
- ModuleInterface
  - SuperAlgorithm+

### :dart: Build targets

- GLAD: Pre-compilation of `glad.c` for source and GL-context sharing among build targets.
- SuperTerrain+: Main procedural terrain generation engine.
  - SuperAlgorithm+: A library of useful noise algorithms for pipeline programming.
    - SuperAlgorithm+Host: Algorithms and prep-works for device computation that are best suited for CPU execution.
    - SuperAlgorithm+Device:  Performant algorithms that can be benefitted from parallel compute.
- SuperDemo+: An application which demostrates the usage of the main engine.
- SuperTest+: Unit test program for dynamic testings.

### :ballot_box_with_check: CMake options

| Option | Explaination |
| ------ | ----- |
| STP_CUDA_RUNTIME_LIBRARY | Set the global `nvcc` compiler flag `-cudart=` to the value set |
| STP_CUDA_VERBOSE_PTX | Append to the global `nvcc` compiler flag with `--ptxas-options=-v` |
| STP_USE_AVX2 | Use AVX2 instruction sets on all vector operations |
| STP_BUILD_DEMO | Enable build SuperDemo+ program |
| STP_BUILD_TEST | Enable build SuperTest+ program |

## :bookmark_tabs: Getting Started

### :desktop_computer: Prerequisites

**Hardware requirement**

- CPU: Intel i5-8400/AMD r5-2400G
- GPU-RT: Nvidia RTX 2060

> Supported compute capability >= 7.5. Nvidia Optix ray-tracing enabled.

- GPU-Legacy: Nvidia GTX 1050

> Supported compute capability >= 6.1. Intel/AMD not available.

- RAM: 4GB
- OS: x64 Windows 10/Linux

**Software requirement**

- OpenGL 4.5 compatible renderer
- CUDA 11.2 with NVRTC (Nvidia Runtime Complication)
- CMake 3.18

### :gear: Installation

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

4. Aquire project build script from CMake.

```sh
cmake ../
```

5. Configure CMake cache if preferred. Leave it as default otherwise.

6. Build the program

```sh
cmake --build ./ --target ALL_BUILD
```

Executables:

- `SuperDemo+` if demo build is enabled
- `SuperTest+` if test build is enabled

## :world_map: Road Map

### [Project Board](https://github.com/stephen-hqxu/superterrainplus/projects)

### [Release Note](https://github.com/stephen-hqxu/superterrainplus/releases)

## :seedling: Contribution

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

## :page_facing_up: Lincense

Distributed under the MIT License. See `LICENSE` for more information.

## :telephone: Contact

Stephen Xu - stephen.hqxu@gmail.com

Project Repository: https://github.com/stephen-hqxu/superterrainplus

## :books: Reference

- [README template](https://github.com/othneildrew/Best-README-Template)
- [Ported from Minecraft biome generator](https://github.com/KaptainWutax/BiomeUtils)
- [Particle based hydraulic erosion](https://github.com/SebLague/Hydraulic-Erosion/tree/Coding-Adventure-E01)