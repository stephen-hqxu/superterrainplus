# Release 0.7.8 - Project Restructure and Exception Handling

## Resolution for issues

### Issue #23

- Add `STPSmartDeviceMemory` in `SuperTerrain+` Utility. Replace all memory that are previously freed manually with `cudaFree()` with the new smart device memory.
- Move pointer assignment to `STPSmartDeviceMemory` to right after `cudaMalloc()`, if memory allocation fails no free is required, or if future operation throws an exception the memory will be freed automatically.
- Implement `std::unique_ptr` for `nvrtcProgram`, `CUmodule` and `CUlinkState` in `STPDiversityGeneratorRTC`, and replace all occurrences with managed memory.
- Improve exception handling in `STPChunkProvider`. All exceptions from async compute threads are captured into an exception queue and will be merged together and rethrow to `STPChunkManager`.
- Remove try-catch blocks in `STPHeightfieldGenerator`, `STPFreeSlipGenerator` and `STPFreeSlipTextureBuffer`. Use `std::unique_ptr` with custom deleter to recycle memory so no memory leak is introduced when exception is thrown.

### Issue #24

Files that were in `CoreInterface` are now moved into a newly created folder `CoreInterface/SuperTerrain+`. This is a more standard way of serving libraries to allow user to distinguishing between a mixture of different libraries and avoid include name collision.

Change in include directory structure has been addressed in all sources and headers.

- Simplify `STPDeviceErrorHandler`. Remove error severity switch due to its unnecessarily. It now throws exception by default and un-caught exception or exception thrown in the destructor will simply terminate the program.
- Change setting macro for `STPDeviceErrorHandler` from `STP_SUPPRESS_MESSAGE` to `STP_DEVICE_ERROR_SUPPRESS_CERR`. This will prevent error handler from writing exception message to `std::cerr` stream.

## General fixes and improvement

- Replace the following external libraries with imported package, and remove them from the External directory:
  - SglToolkit
- Simplify template instantiation model in `STPMemoryPool`.
- Remove `using` and unnecessary include from `STPFreeSlipManager.inl`.
- Declare `STPEngineInitialiser` as `final`.