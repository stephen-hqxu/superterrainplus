# Release 0.2.3 - Preparing for Biome Generator

## Highlights

- Reduce the dynamic memory usage.
- More semantics improvement from the last updates.
- More test cases implemented.

## Updates to biome generator

- Implement multi-threaded biome map generator with reusable caching.

> I finally found a way to implement biome generator more efficiently in multi-threaded environment by splitting generations into different threads and assigning each worker its own layer cache.

- Implement more biome layers for demo.
- Engine-breaking bug fixes.

## Memory pool (#9)

- Memory pool has fully customisable allocator and deallocator, and user-defined arguments.
- Implement memory pool for GPU compute cache on heightfield generator to improve performance.
- Implement memory pool for biome map generator to reuse biome cache for its multithreaded environment.

## Fixes

- Fixing some potential race conditions for the memory pool.