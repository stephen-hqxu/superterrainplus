# Release 0.6.3 - Prepare for Biome Interpolation

## Major changes to STPSingleHistogramFilter

- Move `default` declared destructor for `STPSingleHistogramFilter` into source code so compiler won't complain about using incomplete type of a pointer to implementation.
- Remove constructor of `STPSingleHistogramFilter` to predict the size of the texture because it won't make any difference to the performance. Now `STPSingleHistogramFilter` is trivially constructable.
- Augment the data structure, now the output histogram is no longer bound to a particular STPSingleHistogramFilter instance, but rather, user can ask for a single histogram buffer where the output will then be stored.
- Split `STPSingleHistogramFilter::STPHistogramBuffer` with two versions, being default-allocated and page-locked-allocated. The buffer passed to external will always be using page-locked allocator for fast device memory transfer.

## General fix and improvement

- Refactor code base using new features in C++17 like conditional initialiser and structured binding,
- Enable move constructor/operator in `STPChunk` and `STPChunkStorage`.
- Simplify class construction using initialiser list, for `STPRainDrop`, `STPSingleHistogramFilter` and `STPFreeSlipGenerator` and various `STPEnvironment` classes.
- `STPChunk` will now always allocate memory for internal texture during construction.
- Fix some typos.
- Fix a detrimental bug that causes host memory to be freed before CUDA stream has finished execution. To resolve that we use CUDA stream host callback.
- Augment `STPPinnedMemoryPool` such that it can be used on pageable memory. To avoid confusion it has been renamed to `STPMemoryPool` and use template parameter to denote the memory type of it.

### Partial resolution to issue #21

- Use `remove_pointer` to determine the base type for `cudaStream_t` instead of hard coding.
- Change the storage type in `STPChunkStorage` from `unique_ptr<STPChunk>` to `STPChunk`.
- Simplify chained function call to set compiler and linker flag in `STPDiversityGeneratorRTC`.