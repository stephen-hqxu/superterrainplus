# Release 0.12.13 - Bug Fix and Optimisation

## General fixes and improvement

- Rearrange GLAD build target file structure so it matches the default file structure downloaded via [GLAD web generator](https://glad.dav1d.de/). You may need to clean and re-compile the entire build tree.
- Replace all uses of `std::reduce` with `std::accumulate` for the following class because it is not necessary to use so when `std::execution` is not used, also some compilers have yet support this function:
  - `STPSingleHistogramFilter`
  - `STPCascadedShadowMap`
  - `STPGaussianFilter`

### Chunk coordinate

Re-specification of chunk coordinate system. Chunks are now located using an integer vector of chunk coordinate in world position, the chunk coordinate will be a multiple of chunk size.

The benefit of using integer values effectively avoids floating point errors when looking up chunks in the hash table using chunk coordinate as key.

### Terrain erosion

- Make get erosion brush function in `STPRainDropSetting` returns pointer to constant value, was pointer to value.
- Improve erosion implementation in general.
- Potentially resolve #39 and fix a race condition in the random number generator by assigning a different RNG to each generation invocation and pool the RNG.