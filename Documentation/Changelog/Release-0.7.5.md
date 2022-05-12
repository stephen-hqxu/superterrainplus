# Release 0.7.5 - Error Handling and Build System Change

## SuperTest+

- Fix an integer underflows issue when line is wider than the console width in `STPConsoleReporter`.
- Add a smart text wrapper for the console reporter.
- Add test for free-slip utilities.

## General fixes and improvement

### Exception handling

- Simplify exception handling in some classes to ensure there's only one try-catch block in one function. If exception can be thrown out safely without the need to clear memory, try-catch block is removed.
- Avoid throwing exception in the destructor, and instead terminate the program.
- Replace `exit(-1)` with `std::terminate()`.
- Add a new exception class `STPInvalidEnvironment` which is thrown when `STPEnvironment` is not validated at the constructors of `STPChunkProvider`, `STPHeightfieldGenerator` and `STPermutationGenerator`.
- Add numeric range check for functions in `STPChunk`.
- Add size check in `STPMemoryPool` and `STPFreeSlipGenerator` to make sure the numbers provided are positive.
- Add input value range check for `STPFreeSlipTextureBuffer`.

### File system change

- Move free-slip utilities into a separate folder GPGPU/FreeSlip.
- Re-style include directories. Header should prefer using double quotes while sources prefer using angle brackets. Replace system header includes with angled brackets.
- Replace the following external libraries with imported package, and remove them from the `External` directory:
  - glfw3
  - glm
  - OpenGL
  - SIMPLE

Out final goal is to replace all explicit include directories with CMake imported targets via CMake `find_package()`.
- Change `GLAD` build type to share library. This is beneficial for sharing OpenGL contexts between the main engine and demo application.

### Others

- Remove checker for disabled free-slip hydraulic erosion and return `nullptr` if so. It's generally unsafe and adds more overhead during run-time to check for null pointer.
- Use `emplace_back()` instead of `push_back()` for `getRegion()` in `STPChunk`.
- Address changes brought by `SIMPLE v2.0`.