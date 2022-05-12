# Release 0.7.12 - Test for Algorithm and Runtime Compiler

## Development on SuperTest+

- Add the following test code
  - STPTestPermutation for `STPPermutationGenerator`.
  - STPTestHistogram for `STPSingleHistogramGenerator`.
  - STPTestRTC for `STPDiversityGeneratorRTC`.
- Update `STPConsoleReporter` to address some important bug fixes in the latest version of `Catch2`:

> See https://github.com/catchorg/Catch2/commit/e5938007f7dab5c9886ca561d09106ee4d7f2301 and https://github.com/catchorg/Catch2/commit/290c1b60e6c6d9fd4dde1b28959a6d673caad937 for the details of changes.

## General fixes and improvement

- Replace all `double` with `float` in `STPPermutationGenerator` in host algorithm library and `STPSimplexNoise` in device algorithm library.
- Simplify permutation table generation with `glm`.
- Make the constructor for `STPFreeSlipManager` public and remove `STPFreeSlipGenerator` as its friend class.
- Fix an incorrect pre-launch argument check in `STPSingleHistogramFilter`.
- Add a `cudaFree(0)` call in `STPEngineInitialiser` to enforce context creation at startup.
- Fix an incorrect cache loading operation during rendering buffer generation which causes `int` to be implicitly cast to `unsigned int` and underflow to `UINT32_MAX` instead of -1 as expected.

> Visually it's unaffected when free-slip erosion is turned on because selective edge copy prevents rendering buffer generator from modifying the edge pixel.

- Adjust the working directory for the test target.
- Case valued variable to `uintptr_t` before casting to `void*` when passing compiler flags.