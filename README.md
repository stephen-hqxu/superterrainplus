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
	<img src="https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white" />
</p>

## :eyes: Overview

Super Terrain + (or STP in short) is a procedural terrain generation engine that incorporates natural physics-simulation and photo-realistic rendering. It was originally my computer science undergraduate project and inspired by games with random generation features, most noticeably Minecraft.

**Design Lemma**

- Realistic
- Procedural
- Pseudorandom generation
- Real-time rendering

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
- [ ] Weather effect
- [ ] Real-time raster-ray tracing hybrid rendering

### Procedural entity geneartion

- [ ] Volumetric cloud rendering
- [ ] Procedural foilage generation and placement

### Procedural 3D (volumetric) infinite terrain

No plan, yet.

## :bricks: Middleware

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
- [SQLite3](https://www.sqlite.org/index.html)

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

- CPU
<p align="left">
	<img src="https://img.shields.io/badge/Intel-Core_i5_8400-0071C5?style=flat&logo=intel&logoColor=white" />
	<img src="https://img.shields.io/badge/AMD-Ryzen_5_2400G-ED1C24?style=flat&logo=amd&logoColor=white" />
</p>

- GPU-Legacy
<p align="left">
	<img src="https://img.shields.io/badge/NVIDIA-GTX_1050-76B900?style=flat&logo=nvidia&logoColor=white" />
</p>

> Supported compute capability >= 6.1. Intel/AMD not available.

- GPU-RT
<p align="left">
	<img src="https://img.shields.io/badge/NVIDIA-RTX_2060-76B900?style=flat&logo=nvidia&logoColor=white" />
</p>

> Supported compute capability >= 7.5. Nvidia Optix ray-tracing enabled.

- RAM: 4GB
- OS
<p align="left">
	<img src="https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black" />
	<img src="https://img.shields.io/badge/Windows_10-0078D6?style=flat&logo=windows&logoColor=white" />
</p>

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