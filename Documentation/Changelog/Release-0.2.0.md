# Release 0.2.0 - Cache Optimisation and Test Framework

## Release highlight

- More types of constructors are implemented according to #3. Some functional classes such as rendering classes that do not expect to be created from existing instance will be given `delete` modifiers, utility classes like settings have been given explicit implementations.

> More constructors will be coming in the future, after the program comes out from alpha state. Now my priority is to finish biome generator.

- Improving programming practice, especially adding `const` for certain functions such that it can be used in constant semantics.
- GPU memory and cache optimisation for constant values and random-read-only lookup tables.
- Improve compiler settings for better debug experience and performance under release mode.
- Adding debug preprocessor `_STP_DEBUG_`.
- Updating minimum system requirement, see *README*.
- Minor bug fixes for biome generators.

## Introduction to test framework

- Adding testing project with test framework.
- Implementing a dedicated test report format for Super Terrain +.
- Implementing tests along with bug fixes (that's why tests are important).

> More test cases are coming...

## Bug fix

- Providing solution for #2, a copy of host class will be kept to retain the underlying device pointers.
- Providing solution for #10, it's yet implemented in the current release but it can be easily workaround by simply reduce the number of thread per block.
- Fixing typo for *README* file.

> We are working on biome generator to our best. Source code for biomes are released together for public debugging and internal testing purposes only.

## Proposed features

- #11 requested a terrain collision first-person camera.