# Release 0.9.0-Preview.2 - Setup Sky Renderer

## SuperRealism+

- Add two utilities, `STPCamera` and an inherited class `STPPerspectiveCamera`.
- Refactor the directory structure and group classes into folders.
- Refactor three shader managers in `SuperRealism+` using pure RAII given greater flexibility.
- Add a simple wrapper `STPDebugCallback` for handling OpenGL debug output automatically.
- Add a simple context manager `STPContextManager` for storing current context state and issues context state change functions whenever necessary.
- Add `STPIndirectCommand` as an indirect rendering command wrapper according to GL specification.

> All wrappers are trivially implemented and only functionalities used within this project are written, so they are not complete for the sake of development time.

### GL object management

- Add more GL object wrappers to `SuperRealism+` library:
  - `STPBuffer` for GL buffer object such as vertex buffer and element buffer.
  - `STPVertexArray` for vertex array buffer object.
- Add more GL compatibility types to `STPOpenGL`.

### Sky renderer

- Add a new setting `STPAtomsphereSetting` for storing atmosphere rendering parameters.
- Setup shader `STPSun` for physically-based atmosphere scattering.
- Setup rendering pass for `STPSun`.

## General fixes and improvement

- Add a new exception `STPGLError` to the main engine.
- CMake configuration file include directory overhaul. Configuration files are now all placed into different directories distinguished by configuration such that `#include` in source code can remain unchanged when build configuration changes.
  - As a refactor, configuration template is placed into a separate interface target.
- Add a simple file IO utility `STPFile` for reading all contents in the file quickly.
- Fix an issue where `STPNullablePrimitive` cannot be assigned to `nullptr`.