# Release 0.7.0 - New Unit Test System

## SuperTest+

- Upgrade test framework to `catch2 v3.0`.
- Setup test utilities with `CTest` and `catch2` integration.
- Re-program the default test reporter, `STPConsoleReporter`.
- Add new test units:
  - STPTest2DChunk
  - STPTestUtility

## Custom exception class

- Add a new namespace `STPException`, located in `CoreInterface/Utility/Exception`.
- Add new exception classes:
  - STPBadNumericRange
  - STPCompilationError
  - STPCUDAError
  - STPDeadThreadPool
  - STPInvalidArgument
  - STPMemoryError
  - STPSerialisationError
  - STPUnsupportedFunctionality
- Replace all standard-library exception in the engine with custom exception. This makes catching more specific exception during unit testing easier.

## General fixes, improvement and refactoring

- Simplify formula for generating the global-local index table.
- Replace all CUDA vector functions (uint2, float3, etc.) with `glm` vectors (uvec2, vec3, etc.) and simplify calculations using said vector functions.
- Remove all unnecessary use of `floor()` on unsigned integer division.
- Simplify codes with glm functions where applicable.
- Refactor `STPThreadPool` to cut down the number of unnecessary code.