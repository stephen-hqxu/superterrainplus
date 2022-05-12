# Release 0.3.0-Preview.1 - Basic Free-slip Hydraulic Erosion

## Basic feature with free-slip hydraulic erosion (#1)

- Implementing global-local index converter. We are using a pre-computed index table based on the INI parameters.
- As revised in the previous version, heightmap generation is now split into configurable stages. Chunk compute and loading process are now separated into different functions.
- Improve performance of sparse data copy between host and device.

> For more development discussions, please refer to the issue page :point_right:  https://github.com/stephen-hqxu/superterrainplus/issues/1#issuecomment-814930153

### Issues and "not yet implemented" with the current preview

- Data races identified, the same chunk is modified by different threads at the same time, causing some chunks not computed correctly. For a temporary workaround, multithreading is disabled.
- Erosion brush calculation is not correct with free-slip range.
- Normal map and formatting are not updated after eroding the central chunk, such that chunks are not seamless.
- Though not noticeable on the generated terrain, erosion brush indices are no longer correct under neighbour chunk logic.
- Normal map is not seamless.

### Possible Improvement

- User should be given an option to disable free-slip hydraulic erosion. To disable this in the current build, set free slip chunk range to (1,1). However this doesn't effectively prevent the generation of global-local index table, even though global and local index are the same.
- Thread model looks excessively complicated.

## Other improvements

- Improving documentation of how to configure and use the free-slip erosion system.