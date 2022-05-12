# Release 0.4.4 - Improve Level of Generality

In this release we aim to remove heightmap generation stage from the engine, and instead we take in the user-defined `STPDiversityGenerator` and call the virtual function for generation.

## Programmable pipeline stage

- Rename `STPDiversityGenerator` to `STPDiversityGeneratorRTC`. And add a new class `STPDiversityGenerator`. For runtime-compiled heightmap generation script `STPDiversityGeneratorRTC` can be used; for static-compiled heightmap generation script `STPDiversityGenerator` can be use instead. This can help cutting down the class size if runtime complication is not required, or not supported on target machine.
- Allow input type to be customised in `linkProgram()` in `STPDiversityGeneratorRTC` for better control over complier and linker flags.
- Allow attaching archive to `STPDiversityGeneratorRTC`.
- Main generator `STPHeightfieldGenerator` no longer takes simplex noise to generate heightmap, instead it uses `STPDiversityGenerator` implemented by user.
- Overhaul to `STPSettings` and INI to reduce redundancy for the new system.
- Move `STPBiome` and `STPBiomeSettings` into namespace `STPDemo`. Also they are moved to *SuperTerrain+\World\Biome\Biomes*.
- Move all heightmap generation algorithms to `STPDemo`, and it's located in `STPMultiHeightGenerator` whereas `STPBiomefieldGenerator` implements `STPDiversityGenerator`. The demo generation uses static complication.

## Improvement and fixes

### Suggestions mentioned in #19

- [x] Correct the include guard for `STPBiomeFactory` so the symbol is consistent with the filename.
- [x] Include function info for `STPDeviceErrorHandler`.
- [x] Catch and rethrow exception thrown in `STPDiversityGeneratorRTC` instead of letting the program to crash.
- [ ] ~~Put in-class structure declarations in a separate inline file in `STPDiversityGeneratorRTC`.~~

### General

- Rename `STPSettings` to `STPSetting` to resolve naming conflict with namespace `STPSettings`.
- Remove the following settings from `STPHeightfieldSettings`, to address the fact that heightmap generation is now user-defined.

```cpp

float Scale;
unsigned int Octave;
float Persistence;
float Lacunarity;

```

- Improve exception catching and handling.