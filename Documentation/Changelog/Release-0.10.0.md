# Release 0.10.0 - HDR Rendering

## STPRealism

- Introduce `STPFrameBuffer` and `STPRenderBuffer` as smart managed OpenGL objects.
- Add `textureStorageMultisample()` method is `STPTexture`.
- Add a new managed object `STPSampler`.
- Refactor `STPTexture` and extract functions which set texture parameters to a new interface class `STPImageParameter`. `STPSampler` also derives from this interface class.

### STPPostProcessing

An off-screen renderer that captures rendering buffer from other renderers and process the buffer before drawing it onto the screen.

- Allow attaching multi-sampled texture.
- The shader provides the following functionalities:
  - HDR tone mapping, specifically it supports the following tone mapping functions:
    - Gran Turismo
    - Lottes
    - Uncharted2
  - Gamma correction.

Also enable HDR tone mapping and gamma correction in the demo program.

## General fixes and improvement

- Update `Catch2` to *v3.0.0-preview4*.
- Fix a few typos in the README document.
- Fix a bug for which GL reports invalid image format when `unbindImage()` is called in `STPTexture`.
- Replace `std::shared_mutex` with `std::mutex` in `STPSingleHistogramFilter`. Also eliminate unused function arguments.
- Improve shader source macro definition, add a helper class to convert other data type to string automatically.
- For the demo program, albedo texture are now corrected to linear space, default framebuffer has multi-sampling disabled.
- Change `traverse()` in `STPScenePipeline` to a template function and use bit flag to determine components to be rendered in compile time to avoid having a lot of branches in runtime.