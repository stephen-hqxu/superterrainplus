# Release 0.2.5 - Implementing Suggested Features

## Highlights

- Refactor `STPChunkProvider` and `STPHeightfieldGenerator`. `STPImageConverter` has been removed. Now all heightfield generations are performed step-wise using flags, meaning programmers can have their own combination of generation procedures instead of forcing them to generate everything that they may not need. This also provide benefits to break the system down into modules and improve maintainability.

### Chunk serialisation (#7)

- `STPChunk` can output and input chunk data into and from stream.
- Test program implemented.
- Currently the program doesn't make use of this functionality. Later the program will selectively serialise chunks to secondary storage when it reaches the upper memory bound.

### Free-slip hydraulic erosion (#1)

- We are working on this suggested feature right now, and the pre-release version should be published soon.

## Fixes and improvements

- Fixing precision problems for the random number generator in biome layer.
- Fixing GPU side crash due to the removal of `floor` function after some updates of Visual Studio 2019, as being reported in issue #14.
- Reducing usage of raw pointers.
- Reducing unnecessary thread usage.
- Improving error handling, program will throw errors when chunk compute cannot be completed.
- Removing deprecated documentation.
- Fixing illegal memory access.

## Plan

In version 1.0.0, Super Terrain + will be capable of generating multi-biome terrain, and will be more stable than the current version.
It should be coming soon, we are looking forward to releasing alpha version in the near future.
Later, we will implement more terrain generation algorithm, see repo project page.