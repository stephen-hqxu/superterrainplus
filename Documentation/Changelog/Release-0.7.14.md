# Release 0.7.14 - Coverage and Prototype Testing

## Development on SuperTest+

- Add more test cases to `STPDiversityGeneratorRTC` to cover more scenarios.
- Add missing test scenarios in `STPFreeSlipTextureBuffer`.
- Add missing tests for, in the existing test source code:
  - `STPChunkStorage` in `STPTestChunk`.
  - `STPSmartDeviceMemory` in `STPTestUtility`.

### Prototype test

`STPPrototypeCachedConvolution` for rendering buffer generation in `STPHeightfieldGenerator`. It aims to test if the rendering buffer is the same as before and after shared memory optimisation.

### Coverage test

We perform coverage test for the program to ensure our test cases cover every line of code.

Functionalities added to test to ensure full coverage:
- Rendering buffer retrieval in `STPChunk`.
- Exception when asking for volumetric biomemap and memory pool in `STPBiomeFactory`.
- Exception when requested memory size is zero in `STPMemoryPool`.
- Exception when inserting into a dead thread pool in `STPThreadPool`.
- Exception when asking for memory location when texture buffer is unmerged in `STPFreeSlipTextureBuffer`.
- Storage clear in `STPChunkStorage`.
- Individual data option in `STPDiversityGeneratorRTC`.
- Stream created with priority in `STPSmartStream`.

## General fixes and improvement

- Rename member functions in `STPChunkStorage` for more consistent style. Add more auxiliary functions.
- Elaborate the documentation in `STPFreeSlipTextureBuffer` to notify the user about the memory behaviour of host memory when the texture buffer is in device mode.
- Improve index clamping in `STPHeightfieldGenerator` with `glm` functions and refactor filter kernel.
- Remove conditional instantiation for rendering buffer in `STPFreeSlipGenerator`, instead a static assertion is added.
- Improve documentation for `STPSmartStream` about the stream priority, and flip the output priority to `[greatest, least]` instead of using `[low, high]` as CUDA defines *low* as *greatest*, which was not the program was assuming previously.

### Engine breaking bug in STPLayer

- Fix an incorrect use of local static variable in `STPLocalRNG` which causes the local seed for every RNG instance to be the same, as well as data racing.
- Remove `floorMod()` function in `nextVal()` and use modulo operator instead, they work the same way when the input data have the same sign.

> The same changes have been applied to `STPVoronoiLayer` as well.

- `STPLocalRNG` takes in a layer seed in the constructor, and uses layer seed to mix the local seed.