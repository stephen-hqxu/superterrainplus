# Release 0.6.0-Preview.2 - Free-Slip Texture Buffering

## STPFreeSlipTextureBuffer

STPFreeSlipTextureBuffer is a memory manager dedicated for managing texture from free-slip neighbour chunks and merging them into a large local-indexed buffer for free-slip computing; like what `STPHeightfieldGenerator` does previously.

- `STPFreeSlipTextureBuffer` can now merge buffer to different memory space (host and device) depends on caller.
- Deprecate memory management in `STPHeightfieldGenerator`.
- Improve level of abstraction in `STPFreeSlipGenerator`, which now takes a pointer to `STPFreeSlipTextureBuffer` instead of a pointer to texture.
- Improve documentation in `STPFreeSlipGenerator::STPFreeSlipAdaptor`.
- Improve behaviour for STPFreeSlipTextureBuffer when doing repetitive calls to retrieve texture buffer. See documentation for more details.

### Future work

- Allocation of pinned memory in `STPFreeSlipManagerAdaptor` is not pooled currently.

## Development on STPSingleHistogramFilter

- We finally device to swap the order of two separate filter stages in `STPSingleHistogramFilter` as benchmark shows there's negligible performance difference as the original one. The output histogram from calling `STPSingleHistogramFilter` is now in row-major order.
- Correct the documentation in `STPSingleHistogramFilter` so it matches the new algorithm.
- Extract `STPSingleHistogramFilter::STPFilterReport` as a separate structure and rename it to `STPSingleHistogram`.
- Add `STPSingleHistogramWrapper` in `SuperAlgorithm+Device` for easy `STPSingleHistogram` access on GPU.

## General fixes and improvement

- Rename `destroy()` in `STPSingleHistogramFilter` to `destroyReport()` to avoid confusion.
- Inline more in-source functions in `STPSingleHistogramFilter`.
- Change the argument in filter execution call in `STPSingleHistogramFilter` to `STPFreeSlipManagerAdaptor` for better abstraction.
- Make `STPFreeSlipManagerAdaptor` into a template class to improve type safety.
- Improve hashing algorithm for `vec2` type in `STPChunkStorage`.
- Remove shared library export symbols for `enum` and POD-structure.