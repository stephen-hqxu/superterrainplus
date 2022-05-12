# Release 0.8.0-Preview.2 - Texture Data Structure Generation

## Texture utilities

- Update the hashing algorithm for key pair using the newly introduced `STPHashCombine`.
- Split texture data structures into a separate file `STPTextureInformation`.
- Refactor `STPTextureSplatBuilder` with `STPTextureInformation`.
- Move `STPTextureSplatBuilder` into `STPTextureDatabase`.
- Simplify texture group logic in `STPTextureDatabase`. Now texture group only stores information about the texture format, texture data is mapped to texture ID type mapping table.
- Texture ID is now assigned by the system rather than the user. Also rename various functions to make sure all operations make sense.
- `STPTextureFactory` can now convert texture database into arrays that can be used by OpenGL.
- Create a `STPDatabaseView` which allows querying large result sets from database.

### Database system

- Setup SQLite3 database. Add database initialiser in `STPEngineInitialiser`. Add helper header `STPSQLite`.
- Overhaul `STPTextureDatabase` and `STPTextureSplatBuilder` with in-memory private SQL database.
- Add a new exception type `STPDatabaseError`.
- Allow `STPDeviceErrorHandler` to handle database error.

## General fixes and improvement

- Simplify `STPHashCombine`, it now contains only the variadic version of `combine()`.
- Latest stable version of test framework has been updated to https://github.com/catchorg/Catch2/commit/48a889859bca45ee2c5e5064199c1e5b4b3e00cb
- Use namespace instead of class for which only contain static function and object creation is not intended.

### Improvement to STPSingleHistogramFilter

- Refine the algorithm for removing empty bin. Instead of unnecessarily looping through the whole dictionary, only dictionary entries affected due to removal of bin will be visited and updated. Approximately 20% speed up compared to the old algorithm.