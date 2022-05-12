# Release 0.7.17 - Finalise Test Program

## Changes made in STPRainDropSetting

- Add a new class `STPErosionBrushGenerator` which is responsible for generating the erosion brush and managing the memory space.
- Implement a PIMPL for erosion brush generation. The implementation is defined in `STPErosionBrushGenerator`. This is for avoiding sending useless objects like `std::vector` (takes up 40 bytes each) to the device.
- Explicitly declare the following classes as non-copyable to make sure the erosion brush cache on host side is unique:
  - STPRainDropSetting
  - STPHeightfieldSetting
  - STPConfiguration
- Replace raw device pointer with `STPSmartDeviceMemory`.
- Change the usage in `SuperDemo+` so setting objects are moved instead of copied.

## Development on SuperTest+

- Add `STPTestInformation` to store common data for testing.
- Add stream-ordered memory allocation testing for `STPSmartDeviceMemory`.
- Add device memory test for `STPFreeSlipTextureBuffer` and `STPFreeSlipGenerator`.
- Add `STPTestEnvironment` for testing class under namespace `STPEnvironment`.

By now, all classes except the following are tested with theoretical full coverage:
- STPChunkManager
- STPChunkProvider
- STPHeightfieldGenerator (only tested using prototype)
- STPRainDrop
- STPSimplexNoiseGenerator

The objects above are mostly in our generator pipeline stages, testing involves a large amount of work that can even surpass the effort to program the engine itself. We are going to either do the test using different methods (e.g. static or user acceptance test), or do it in the unit test manner but defer it to later. There are still awaiting goals to be completed by deadline.

## General fixes and improvement

- Declare `constexpr` and `static` or both for lambda expression instantiation whenever necessary.
- Remove some unnecessary *pass by reference*, for example `glm` vector types which are usually small and cheap to copy than dereferencing.
- Simplify formula in `STPVoronoiLayer`.
- Update cache load algorithm in rendering buffer generator to optimised for coalesced memory access.
- Remove inheritance from `STPFreeSlipData` in `STPFreeSlipGenerator` because it doesn't make sense.
- Improve `STPChunkSetting` validation.