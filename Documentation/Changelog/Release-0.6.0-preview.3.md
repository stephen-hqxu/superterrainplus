# Release 0.6.0-Preview.3 - New Utilities

## STPSmartStream

- Add a smart CUDA stream manager that utilises RAII techniques -- create a stream at construction, destroy the stream at destruction.
- Implement smart stream from `unique_ptr` such that the object can only be moved.
- Implement custom cast operator to cast `STPSmartStream` to `cudaStream_t` implicitly so no codebase needs to be changed.

## STPPinnedMemoryPool

- Implement a new memory pool system dedicated for pinned memory.
- Memory is managed by smart pointer.
- Implement memory header which contains information about the allocated memory while external user is completely unaware of that, which makes returning memory back to the pool easier.
- Make pinned memory pool thread safe.

## General fixes and improvement

- Deprecate `STPMemoryPool` in `SuperTerrain+` Utility.
- `STPFreeSlipTextureBuffer` now uses `STPPinnedMemoryPool` for pinned memory allocation.
- Change the argument in filter execution call in `STPSingleHistogramFilter` to `STPFreeSlipManager` for data safety.