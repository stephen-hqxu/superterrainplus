# GL Global Buffer Binding Index Reservation

*SuperRealism+* uses shader storage buffer to store data shared by multiple renderers to cut down the amount of binding, such that certain bindings are reserved for internal usage, and user application should not occupy the following binding indices.

| Binding | Definition | Usage |
| ------- | ---------- | ----- |
| 0 | STPCameraInformation | View space transformation vectors and matrices |
| 1 | STPLightSpaceInformation | Light space transformation data for shadow rendering |
| 2 | STPMaterialRegistry | Material properties for some special types of object |
| 3 | STPRayTracedIntersectionData | Geometry data from ray traced screen space intersection shader |
| 4 - 9 | (Unused) | Reserved for future implementation |