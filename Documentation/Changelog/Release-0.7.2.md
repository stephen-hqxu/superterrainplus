# Release 0.7.2 - Testing and Improvement

## More test units in SuperTest+

- Complete `STPTestUtility`, the test for utilities.
- Refine on the test cases to make it more clear.
- Add new test `STPTest2DBiome` for biome generation testing.

## General fixes and improvement

A number of bugs are found during writing and running the test program.

### STPDeviceErrorHandler

- Remove unnecessary semi-colon for the macro defined in `STPDeviceErrorHandler`.
- Refactor `STPDeviceErrorHandler` and `STPFreeSlipGenerator` to eliminate repetitive coding during template explicit instantiation with macros.
- Add a define symbol `STP_SUPPRESS_ERROR_MESSAGE` to `STPDeviceErrorHandler` to suppress printing of any error message.

### Diversity

- Remove unnecessary destructor implementation in `STPLayerCache`.
- The clear value for key has been changed to UINT64_MAX (equivalent to -1 for signed integer system) instead of 0 when layer cache is initialised and cleared in `STPLayerCache`.
- Remove layer cache for the starting layer in `SuperDemo+` since `STPBiomeFactory` only calls the uncached sampling method for the starting layer and it's unnecessary to cache data that only uses once.
- Fix a bug when layer manager is not reused by `STPBiomeFactory`.

### Multi-threading

- Fix an issue in thread pool that causes program deadlock on program exit. This is caused by the looped check in `STPChunk` destructor which will not return until in-use flag is false, and thread pool is killed as soon as it finishes the current task even if the task queue is not empty. If any thread left in the task queue is responsible for releasing the flag for the chunk, it will be a deadlock. A simple fix is don't terminate the thread poll until kill signal is sent **and** the task queue is not empty.
  - Add additional protection (also good programming practice in general) in `STPMasterRenderer` to make sure thread pool is the last object to be destroyed so other stuff can be finished.
  - Add more protection to `STPChunkProvider` to make sure it synchronises and waits for its internal thread pool to finish before getting destroyed.
  - Add even more protection in `STPWorldManager` to make sure `STPChunkProvider` is the first member to be destroyed before destroying other generators.