# Release 0.6.5 - Biome Edge Interpolation

## Data structure in STPSingleHistogramFilter

Profiling shows the implementation using `std::vector` has led to significant performance issue on MSVC compiler when debug runtime library (`/MDd` and `/MTd`) is enabled. Although the final performance in release mode is not affected, the slowness makes debugging during development impossible.

**Result of a run with `std::vector`:**

| Compiler | -O0/ms | -O3/ms |
| --------- | ---- | ---- |
| MSVC 2019 | >2000 | ~60 |
| GCC 10 | ~200 | ~30 |

We also notice time taken in release mode on MSVC is twice as long compared to GCC.

### Improvement

- Implement a custom version of `std::vector`, namely `STPArrayList`. It's a minimal data structure that only serves to `STPSingleHistogramFilter`. The goal is make it as simple as possible.
- Remove all uses of `back_inserter` and instead pre-resize the array list and do a simple copy at once.

**Result of a run with `STPArrayList`:**

| Compiler | -O0/ms | -O3/ms |
| --------- | ---- | ---- |
| MSVC 2019 && GCC 10 | ~140 | ~25 |

### Data sets used in all benchmarks

| Item | Value |
| ----- | ----- |
| Dimension | 512 * 512 |
| Free-slip range | 1536 * 1536 |
| Radius | 32 |
| Sample range | RNG, 0 to 14, both inclusive |

## Implement smooth biome transition

- Implement procedures in `STPBiomefieldGenerator` to enable smooth biome transition with `STPSingleHistogramFilter`. This includes:
  - A CUDA memory pool.
  - A histogram buffer pool for reusing memory.
  - A host memory pool for releasing histogram buffer after execution.
  - Critical sections for thread safety.
  - Stream-ordered operations.
- Making changes in the runtime compiled script to utilises histogram being sent to GPU.
- Pinned memory allocation for histogram buffer is temporarily disabled due to some context issues.

## General fixes and improvement

- Inline definitions of `STPFreeSlipManager` since device functions cannot be exported as shared library.
- Declare `default` on move constructor/assignment operators for certain class to make sure compiler does the right thing.
- Replace custom deleter for `STPPinnedHistogramBuffer` with default-constructed structure so `unique_ptr` can be default constructed.
- Improve documentation on the memory behaviour about the texture buffer.
- Fix an issue causes uncatched compilation error in `SuperDemo+` when using CUDA runtime compiler due to the changes made to the exception throw system previously.
- Remove `const` specifier for variables in `STPSingleHistogram`.
- Change the function parameter from taking a value to a r-value reference in `operator()` in `STPSingleHistogramWrapper`