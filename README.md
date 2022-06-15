![SuperTerrain+ Project Logo](https://user-images.githubusercontent.com/77457215/170828638-faba7769-1a04-46ac-a625-8179914284d1.png)

[![GitHub Releases](https://img.shields.io/github/v/release/stephen-hqxu/superterrainplus?include_prereleases&label=Release)](https://github.com/stephen-hqxu/superterrainplus/releases)
[![GitHub License](https://img.shields.io/github/license/stephen-hqxu/superterrainplus?label=License)](https://github.com/stephen-hqxu/superterrainplus/blob/master/LICENSE)
[![3rd-Party License](https://img.shields.io/badge/License-3rd--party-green)](https://github.com/stephen-hqxu/superterrainplus/blob/master/3rd-Party)
[![Project Roadmap](https://img.shields.io/badge/Project-Roadmap-cd853f)](https://github.com/stephen-hqxu/superterrainplus/projects)
[![Documentation](https://img.shields.io/badge/-Documentation-fa8072)](https://github.com/stephen-hqxu/superterrainplus/blob/master/Documentation/README.md)

## :eyes: Overview

<p align="center">
	<img src="https://img.shields.io/badge/C%2B%2B_17-00599C?style=flat&logo=c%2B%2B&logoColor=white" />
	<img src="https://img.shields.io/badge/NVIDIA-CUDA_11.3-76B900?style=flat&logo=nvidia&logoColor=white" />
	<img src="https://img.shields.io/badge/OpenGL_4.6-FFFFFF?style=flat&logo=opengl" />
	<img src="https://img.shields.io/badge/CMake_3.18-064F8C?style=flat&logo=cmake&logoColor=white" />
</p>

Procedural technique is one of the most popular topics in computer graphics and allows us to generate data using the power of algorithms and minimise efforts spent on editing.

*SuperTerrain+* is a procedural terrain engine and real-time renderer started as my personal project in July, 2020 and later become my undergraduate dissertation project; now I mainly use it as a playground for exploring this area further and improving my programming proficiency; more importantly, for fun.

*SuperTerrain+* provides a low-overhead, extendible and flexible interface, demonstrates application of modern programming and rendering techniques; with the help of extensive in-source documentations, this is also a great place for learning.

## :bulb: Main Features

### Procedural heightfield infinite terrain

- [x] Improved simplex noise
- [x] Hardware tessellation
- [x] Continuous level-of-detail
- [x] Multi-threaded heightfield generation
- [x] Free-slip particle-based hydraulic erosion
- [x] Programmable static/runtime-compiled pipeline
- [x] Biome generation
- [x] Multi-biome heightfield generation with smooth transition
- [x] Rule-based biome-dependent texture splatting with smooth transition
- [x] Adaptive distance-based texture scaling
- [ ] River and lake generation

### Procedural volumetric infinite terrain

- [ ] Come up with a nice plan for this challenging topic

### Procedural geometry generation

- [ ] Volumetric cloud
- [ ] Rule-based geometry placement
- [ ] Procedural animated grassland generation
- [ ] Procedural parameter-based tree generation
- [ ] Procedural rock generation

### Real-time photorealistic rendering

- [x] Procedural sky rendering
  - [x] Atmospheric scattering
  - [x] Aerial perspective
  - [x] Seasonal effect
  - [x] Starfield
- [x] Deferred shading
- [ ] Post-processing
  - [x] Filmic tone mapping
  - [x] Gamma correction
  - [ ] Auto-exposure
- [ ] Shadow mapping
  - [ ] Simple shadow mapping for spotlight
  - [ ] Cubemap shadow mapping for point light
  - [x] Cascaded shadow mapping for directional light
- [x] Ambient occlusion
  - [x] Screen-space ambient occlusion
  - [x] Horizon-based ambient occlusion
- [ ] Anti-aliasing
  - [x] ~~Multi-sample anti-aliasing~~
  - [ ] Fast-approximate anti-aliasing
  - [ ] Morphological anti-aliasing
  - [ ] Temporal anti-aliasing
- [ ] Multiple lights
  - [x] Ambient light
  - [x] Directional light
  - [ ] Point light
  - [ ] Spotlight
- [ ] Water rendering
  - [x] Procedural water wave animation
  - [x] Screen-space reflection
  - [x] Screen-space refraction
  - [x] Fresnel effect
  - [ ] Caustics
  - [ ] Underwater crepuscular rays
- [ ] Procedural weather effect
- [ ] Real-time global illumination

### Optimisation technique

- [ ] Frustum culling
- [ ] Variable rate shading
- [ ] Mesh shading
- [ ] Deferred clustered lighting
- [ ] Sparse virtual texture

## :bricks: Middleware

Main engine:

- [GLM](https://github.com/g-truc/glm)
- [GLAD](https://github.com/Dav1dde/glad)
- [SQLite3](https://www.sqlite.org/index.html)

Additional dependencies for the demo application:

- [GLFW](https://github.com/glfw/glfw)
- [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)

Test library:

- [Catch2 v3](https://github.com/catchorg/Catch2)

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

| Option | Note | Default |
| ------ | ---- | ------- |
| STP_CUDA_RUNTIME_LIBRARY | Set the global `nvcc` compiler flag `-cudart=` to the value set | Shared |
| STP_USE_AVX2 | Use AVX2 instruction sets on all vector operations | ON |
| STP_BUILD_DEMO | Enable SuperDemo+ program build | ON |
| STP_DEVELOPMENT_BUILD | Enable development mode | OFF |
| STP_DEVELOPMENT_BUILD::STP_CUDA_VERBOSE_PTX | Append to the global `nvcc` compiler flag with `--ptxas-options=-v` | ON::ON |
| STP_DEVELOPMENT_BUILD::STP_BUILD_TEST | Enable SuperTest+ program build | ON::ON |
| STP_DEVELOPMENT_BUILD::STP_ENABLE_WARNING | Enable all compiler warnings | ON::ON |

Note that `::` denotes a CMake dependent option. Pattern *A::B* default to *valueA::valueB* means option *B* is depended on option *A*, and *B* is exposed in CMake cache set to *valueB* by default if and only if *A* is set to *valueA*; otherwise *B* is hidden from the user and set to *NOT valueB*.

## :bookmark_tabs: Getting Started

### :desktop_computer: Prerequisites

#### Hardware requirement

- CPU

![Intel CPU Requirement](https://img.shields.io/badge/Intel-Core_i5_8400-0071C5?style=flat&logo=intel&logoColor=white)
![AMD CPU Requirement](https://img.shields.io/badge/AMD-Ryzen_5_2400G-ED1C24?style=flat&logo=amd&logoColor=white)

- GPU

![Nvidia GPU Requirement](https://img.shields.io/badge/NVIDIA-GTX_1660-76B900?style=flat&logo=nvidia&logoColor=white)

> *SuperTerrain+* currently relies heavily on CUDA and GL extensions exclusive to Nvidia GPUs.

- OS

![OS Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black)
![OS Windows](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white)

#### OpenGL extension requirement

- OpenGL 4.6 core profile
- [GL_ARB_bindless_texture](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_bindless_texture.txt)
- [GL_ARB_shading_language_include](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shading_language_include.txt)
- [GL_NV_draw_texture](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_draw_texture.txt)
- [GL_NV_gpu_shader5](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_gpu_shader5.txt)
- [GL_NV_shader_buffer_load](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_shader_buffer_load.txt)
- [GL_NV_shader_buffer_store](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_shader_buffer_store.txt)

The following extensions are not required but will be made used by the engine automatically whenever applicable.

- ~~[GL_NV_mesh_shader](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_mesh_shader.txt)~~
- ~~[GL_NV_primitive_shading_rate](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_primitive_shading_rate.txt)~~
- [GL_NV_representative_fragment_test](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_representative_fragment_test.txt)
- ~~[GL_NV_shading_rate_image](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_shading_rate_image.txt)~~

> This is usually not a problem if your GPU meets the minimum system requirement and have a relatively recent driver installed. If you are unsure, we recommend checking with [OpenGL Extensions Viewer](https://www.realtech-vr.com/home/glview) or [online extension database](https://opengl.gpuinfo.org/).

### :gear: Build Instruction

#### External Resources

To reduce the size of repository and keep everything clean, all external materials are not included in the commit.

To simplify the setup process for the first-time build, the following resources are managed automatically by CMake and no additional setup is required. All the rest of the aforementioned dependencies are required to be installed on your computer.

- Assets for the demo program
- stb_image.h
- GLAD

CMake will download the required files hosted on *GitHub Releases*, to directory `STPResourceCache` located at the root of your binary directory. The build system will not re-download them if they already exist; they may get updated in the future at one point, if your older version fails to compile or run, simply deleting the respective directory within `STPResourceCache` can purge the cache and will force a re-download next time the program is built.

#### Build

*SuperTerrain+* uses CMake to build and it behaves similarly to most CMake projects. You may skip these basic instructions if you are familiar to CMake and instead focusing on setting up dependencies and CMake options.

```sh

# use shallow clone to speed up if you only wish to run the demo
# for development purposes please do a blob-less clone by replacing `--depth 1` with `--filter=blob:none`
# avoid full clone due to large assets in the commit history
git clone -b master --depth 1 https://github.com/stephen-hqxu/superterrainplus.git

cd ./superterrainplus

mkdir build
cd ./build

cmake ../

```

*SuperTerrain+* build script automatically searches for dependencies; in case of any error, please make sure you have all third-party libraries installed on your computer. You may want configure *CMakeCache.txt* before moving on to the next step to build the application.

```bat

REM Windows; you can either compile via Visual Studio GUI from the generated VS solution,
REM or alternatively the command line like this
cmake --build ./ --config Release --target ALL_BUILD

```

Or:

```sh

# Unix
make all

```

Compilation of the engine may take up to 5 minutes. You may obtain the following executables:

- `SuperDemo+` if demo build is enabled
- `SuperTest+` if test build is enabled

## :books: Credits

This section contains source code and libraries that are not dependencies of this project but they are where ideas and inspirations are taken from. I will try my best to give attributions and copyright notices for all publications used; in case something is missing, please open an issue.

For redistributed open source project, see *3rd-Party* directory to find the licenses. For academic style *BibTex* references, check out the *Report* directory.

- [README Template](https://github.com/othneildrew/Best-README-Template)

### Terrain generation

- [Particle based hydraulic erosion](https://github.com/SebLague/Hydraulic-Erosion/tree/Coding-Adventure-E01)
- [Minecraft biome generator](https://github.com/KaptainWutax/BiomeUtils)
- [Linear time Gaussian filter](http://blog.ivank.net/fastest-gaussian-blur.html) by *Ivan Kutskir*
- [Stratified sampling technique](https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using) from *GPU Gems 2*
- [Simple C++ lexer](https://gist.github.com/arrieta/1a309138689e09375b90b3b1aa768e20)

### Geometry generation

- [Animated grassland generation](https://github.com/spacejack/terra)

### Photorealistic rendering

- [Normal map blending](https://blog.selfshadow.com/publications/blending-in-detail/)
- [High-level OpenGL function wrapper](https://github.com/cginternals/globjects)
- [Simulating the Colors of the Sky](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky) from *Scratchapixel*
- [Physically-based atmospheric scattering](https://github.com/wwwtyro/glsl-atmosphere/)
- [A Primer On Efficient Rendering Algorithms & Clustered Shading](http://www.aortiz.me/2018/12/21/CG.html) by *Ángel Ortiz*
- [Forward vs Deferred vs Forward+ Rendering with DirectX 11](https://www.3dgep.com/forward-plus/#Experiment_Setup_and_Performance_Results) by *Jeremiah van Oosten*
- [SSAO](https://learnopengl.com/Advanced-Lighting/SSAO) from *Learn OpenGL*
- [HBAO](https://github.com/nvpro-samples/gl_ssao) and [HBAO+](https://github.com/NVIDIAGameWorks/HBAOPlus)
- [PBR theory](https://learnopengl.com/PBR/Theory) from *Learn OpenGL*
- [Procedural water animation](https://www.shadertoy.com/view/4dBcRD)

**Shadow**

- [Cascaded Shadow Mapping](https://learnopengl.com/Guest-Articles/2021/CSM) by *Márton Árbócz* from *Learn OpenGL*
- [Cascaded Shadow Maps](https://docs.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps) from *Microsoft*
- [Variance Shadow Maps](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps)

**Night rendering**

- [Stars and auroras](https://www.shadertoy.com/view/XtGGRt)
- [Star Nest](https://www.shadertoy.com/view/XlfGRj)

**HDR**

- [Filmic tone mapping functions](https://bruop.github.io/tonemapping/) by *Bruno Opsenica*
- [HDR Theory and practice](https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp) by *Hajime Uchimura*
- [Advanced Techniques and Optimization of HDR Color Pipelines](http://32ipi028l5q82yhj72224m8j.wpengine.netdna-cdn.com/wp-content/uploads/2016/03/GdcVdrLottes.pdf) by *Timothy Lottes*
- [Uncharted2: HDR Lighting](http://slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting) by *John Hable*
- [Tone mapping curve sketch](https://www.shadertoy.com/view/WdjSW3)
- [Tone mapping rendering comparison](https://www.shadertoy.com/view/lslGzl)

### Assets

- [AmbientCG](https://ambientcg.com/)
- [Flaticon](https://www.flaticon.com/)