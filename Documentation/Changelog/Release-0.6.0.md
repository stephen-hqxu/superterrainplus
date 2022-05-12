# Release 0.6.0 - Tools for Smooth Biome Transition

In this release we introduces a lot of new algorithms and utilities that are aimed to resolve the issue of strong cut-off between biome edges. The actual implementation for smooth biome transition will be coming later. More development details have been discussed and documented in #22.

## STPSingleHistogramFilter

- Fast execution to generate histograms for every pixel on a discrete texture like biomemap within a given radius. The runtime of the algorithm is loosely depended on the size of the radius which is useful for large scale smooth biome interpolation.
- Shipped with a histogram wrapper for accessing histogram on GPU.
- It's implemented on CPU, located in `SuperAlgorithm+Host` library. The output matrix of histograms is jaggedly flatten and arranged in row-major.

## STPFreeSlipTextureBuffer

It is a wrapper over free-slip texture. Free-slip texture buffer will do the free-slip texture merging and unmerging automatically. User can choose to obtain free-slip texture which is in local-index order, in different memory space, either host or device, flexibly with relatively low overhead.

## STPSmartStream

The smart stream is essentially a wrapper of `cudaStream_t` and `unique_ptr`. A CUDA stream will be created at construction of the smart stream object and auto destroyed at destruction of the smart stream. Like `unique_ptr`, smart stream is only movable.

## STPPinnedMemoryPool

We deprecate the old `STPMemoryPool`, and introduces a more specialised memory pool for pinned memory exclusively with more friendly function call. It also makes use of standard library to make the code more readable.

The new memory pool is thread safe already, and implement a memory header which contains information about the allocation so user doesn't need to tell the memory pool, for example the size of the allocation, when memory is returned.

## General fixes and improvement

- Refactor all classes and make use of the new utilities introduced in this release.
- Improve general programming styles and practice.
- Improvement to hashing algorithm for `uvec2` in `STPChunkStorage`.
- Remove shared library exports for unnecessary classes.
- Make project include directories more consistent such that including each other won't cause file path issues.
- `STPFreeSlipGenerator` can now generate `STPFreeSlipManagerAdaptor` which will generate different `STPFreeSlipManager` for host or/and device usage.
- Pre-cache `STPFreeSlipData` in `STPFreeSlipGenerator` so it can strategically select the data based on chosen memory space.
- Improve various class constructors.
- Replace some throwing-string exceptions with classes inherited from `std::exception`.