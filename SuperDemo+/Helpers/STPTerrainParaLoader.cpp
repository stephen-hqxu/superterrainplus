#include "STPTerrainParaLoader.h"

//Biome Registry, just a demo program
#include "../World/Biomes/STPBiomeRegistry.h"
//System
#include <string>

using namespace STPDemo;
using namespace SuperTerrainPlus;

using STPAlgorithm::STPINIStorageView;
using STPAlgorithm::STPINISectionView;

using std::string;
using std::pair;

using glm::uvec2;
using glm::dvec2;
using glm::vec2;
using glm::vec3;
using glm::dvec3;

STPEnvironment::STPMeshSetting STPTerrainParaLoader::getRenderingSetting(const STPINISectionView& section) {
	STPEnvironment::STPMeshSetting rendering_options;
	STPEnvironment::STPTessellationSetting& tess_options = rendering_options.TessSetting;
	STPEnvironment::STPMeshSetting::STPTextureRegionSmoothSetting& smooth_options = rendering_options.RegionSmoothSetting;
	STPEnvironment::STPMeshSetting::STPTextureScaleDistanceSetting& scale_options = rendering_options.RegionScaleSetting;

	rendering_options.Strength = section.at("strength").to<float>();
	rendering_options.Altitude = section.at("altitude").to<float>();
	
	tess_options.MinTessLevel = section.at("minTess").to<float>();
	tess_options.MaxTessLevel = section.at("maxTess").to<float>();
	tess_options.FurthestTessDistance = section.at("furthestDistance").to<float>();
	tess_options.NearestTessDistance = section.at("nearestDistance").to<float>();

	smooth_options.KernelRadius = section.at("kernel_radius").to<unsigned int>();
	smooth_options.KernelScale = section.at("kernel_scale").to<float>();
	smooth_options.NoiseScale = section.at("noise_scale").to<unsigned int>();

	scale_options.PrimaryFar = section.at("uv_scale_primary_far").to<float>();
	scale_options.SecondaryFar = section.at("uv_scale_secondary_far").to<float>();
	scale_options.TertiaryFar = section.at("uv_scale_tertiary_far").to<float>();

	return rendering_options;
}

STPEnvironment::STPChunkSetting STPTerrainParaLoader::getChunkSetting(const STPINISectionView& section) {
	STPEnvironment::STPChunkSetting chunks_options;

	chunks_options.MapSize = uvec2(
		section.at("heightmap2DSizeX").to<unsigned int>(),
		section.at("heightmap2DSizeZ").to<unsigned int>()
	);
	chunks_options.ChunkSize = uvec2(
		section.at("chunkSizeX").to<unsigned int>(),
		section.at("chunkSizeZ").to<unsigned int>()
	);
	chunks_options.RenderedChunk = uvec2(
		section.at("renderedSizeX").to<unsigned int>(),
		section.at("renderedSizeZ").to<unsigned int>()
	);
	chunks_options.ChunkOffset = dvec3(
		section.at("chunkOffsetX").to<double>(),
		section.at("chunkOffsetY").to<double>(),
		section.at("chunkOffsetZ").to<double>()
	);
	chunks_options.MapOffset = dvec2(
		section.at("mapOffsetX").to<double>(),
		section.at("mapOffsetZ").to<double>()
	);
	chunks_options.FreeSlipChunk = uvec2(
		section.at("freeSlipX").to<unsigned int>(),
		section.at("freeSlipZ").to<unsigned int>()
	);
	chunks_options.ChunkScaling = section.at("chunkScale").to<double>();

	return chunks_options;
}

STPEnvironment::STPHeightfieldSetting STPTerrainParaLoader::getGeneratorSetting(const STPINISectionView& section) {
	//get the default settings
	STPEnvironment::STPHeightfieldSetting launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Seed = section.at("seed").to<STPDiversity::Seed>();
	launch_options.Inertia = section.at("inertia").to<float>();
	launch_options.SedimentCapacityFactor = section.at("sediment_capacity_factor").to<float>();
	launch_options.minSedimentCapacity = section.at("min_sediment_capacity").to<float>();
	launch_options.initWaterVolume = section.at("init_water_volume").to<float>();
	launch_options.minWaterVolume = section.at("min_water_volume").to<float>();
	launch_options.Friction = section.at("friction").to<float>();
	launch_options.initSpeed = section.at("init_speed").to<float>();
	launch_options.ErodeSpeed = section.at("erode_speed").to<float>();
	launch_options.DepositSpeed = section.at("deposit_speed").to<float>();
	launch_options.EvaporateSpeed = section.at("evaporate_speed").to<float>();
	launch_options.Gravity = section.at("gravity").to<float>();
	launch_options.ErosionBrushRadius = section.at("brush_radius").to<unsigned int>();
	launch_options.RainDropCount = section.at("iteration").to<unsigned int>();

	//return the value
	return launch_options;
}

STPEnvironment::STPSimplexNoiseSetting STPTerrainParaLoader::getSimplexSetting(const STPINISectionView& section) {
	auto noise_option = STPEnvironment::STPSimplexNoiseSetting();

	noise_option.Seed = section.at("seed").to<unsigned long long>();
	noise_option.Distribution = section.at("distribution").to<unsigned int>();
	noise_option.Offset = section.at("offset").to<double>();

	return noise_option;
}

pair<STPEnvironment::STPSunSetting, STPEnvironment::STPAtmosphereSetting> STPTerrainParaLoader::getSkySetting(const STPINISectionView& section) {
	STPEnvironment::STPSunSetting sun;
	sun.DayLength = section.at("day_length").to<size_t>();
	sun.DayStartOffset = section.at("day_start").to<size_t>();
	sun.YearLength = section.at("year_length").to<unsigned int>();
	sun.Obliquity = section.at("axial_tilt").to<double>();
	sun.Latitude = section.at("latitude").to<double>();

	STPEnvironment::STPAtmosphereSetting atmo;
	atmo.SunIntensity = section.at("sun_intensity").to<float>();
	atmo.PlanetRadius = section.at("planet_radius").to<float>();
	atmo.AtmosphereRadius = section.at("atmoshpere_radius").to<float>();
	atmo.ViewAltitude = section.at("view_altitude").to<float>();

	atmo.RayleighCoefficient = vec3(
		section.at("rayleigh_coefX").to<float>(),
		section.at("rayleigh_coefY").to<float>(),
		section.at("rayleigh_coefZ").to<float>()
	);
	atmo.MieCoefficient = section.at("mie_coef").to<float>();
	atmo.RayleighScale = section.at("rayleigh_scale").to<float>();
	atmo.MieScale = section.at("mie_scale").to<float>();
	atmo.MieScatteringDirection = section.at("mie_dir").to<float>();

	atmo.PrimaryRayStep = section.at("primary_step").to<unsigned int>();
	atmo.SecondaryRayStep = section.at("secondary_step").to<unsigned int>();

	return pair(sun, atmo);
}

STPEnvironment::STPOcclusionKernelSetting STPTerrainParaLoader::getAOSetting(const STPINISectionView& section) {
	STPEnvironment::STPOcclusionKernelSetting ao_kernel;

	ao_kernel.RandomSampleSeed = section.at("sample_seed").to<unsigned long long>();
	ao_kernel.RotationVectorSize = uvec2(
		section.at("rotation_vector_sizeX").to<unsigned int>(),
		section.at("rotation_vector_sizeY").to<unsigned int>()
	);
	ao_kernel.SampleRadius = section.at("sample_radius").to<float>();
	ao_kernel.Bias = section.at("bias").to<float>();

	return ao_kernel;
}

STPEnvironment::STPWaterSetting STPTerrainParaLoader::getWaterSetting(const STPINISectionView& section, float altitude) {
	STPEnvironment::STPWaterSetting water;
	STPEnvironment::STPTessellationSetting& water_tess = water.WaterMeshTess;
	STPEnvironment::STPWaterSetting::STPWaterWaveSetting& water_wave = water.WaterWave;

	water_wave.InitialRotation = section.at("init_rotation").to<float>();
	water_wave.InitialFrequency = section.at("init_frequency").to<float>();
	water_wave.InitialAmplitude = section.at("init_amplitude").to<float>();
	water_wave.InitialSpeed = section.at("init_speed").to<float>();
	water_wave.OctaveRotation = section.at("octave_rotation").to<float>();
	water_wave.Lacunarity = section.at("lacunarity").to<float>();
	water_wave.Persistence = section.at("persistence").to<float>();
	water_wave.OctaveSpeed = section.at("octave_speed").to<float>();
	water_wave.WaveDrag = section.at("drag").to<float>();

	water_tess.MaxTessLevel = section.at("max_tess").to<float>();
	water_tess.MinTessLevel = section.at("min_tess").to<float>();
	water_tess.FurthestTessDistance = section.at("max_distance").to<float>();
	water_tess.NearestTessDistance = section.at("min_distance").to<float>();

	water.MinimumWaterLevel = section.at("min_level").to<float>();
	water.CullTestSample = section.at("cull_test_sample").to<unsigned int>();
	water.CullTestRadius = section.at("cull_test_radius").to<float>();
	water.Altitude = altitude;
	water.WaveHeight = section.at("wave_height").to<float>();
	
	water.WaterWaveIteration.Geometry = section.at("geometry_iteration").to<unsigned int>();
	water.WaterWaveIteration.Normal = section.at("normal_iteration").to<unsigned int>();
	water.Tint = vec3(
		section.at("tintR").to<float>(),
		section.at("tintG").to<float>(),
		section.at("tintB").to<float>()
	);
	water.NormalEpsilon = section.at("epsilon").to<float>();

	return water;
}

STPEnvironment::STPBidirectionalScatteringSetting STPTerrainParaLoader::getBSDFSetting(const STPINISectionView& section) {
	STPEnvironment::STPBidirectionalScatteringSetting bsdf;

	bsdf.MaxRayDistance = section.at("max_ray_distance").to<float>();
	bsdf.DepthBias = section.at("depth_bias").to<float>();
	bsdf.RayResolution = section.at("ray_resolution").to<unsigned int>();
	bsdf.RayStep = section.at("ray_step").to<unsigned int>();

	return bsdf;
}

void STPTerrainParaLoader::loadBiomeParameters(const STPINIStorageView& biomeini) {
	using namespace STPDiversity;
	namespace BR = STPDemo::STPBiomeRegistry;
	auto load = [&biomeini](STPDemo::STPBiome& biome, string name) -> void {
		const STPINISectionView& curr_biome = biomeini.at(name);

		//assigning props
		biome.Name = curr_biome.at("name");
		biome.ID = curr_biome.at("id").to<Sample>();
		biome.Temperature = curr_biome.at("temperature").to<float>();
		biome.Precipitation = curr_biome.at("precipitation").to<float>();
		biome.Scale = curr_biome.at("scale").to<float>();
		biome.Octave = curr_biome.at("octave").to<unsigned int>();
		biome.Persistence = curr_biome.at("persistence").to<float>();
		biome.Lacunarity = curr_biome.at("lacunarity").to<float>();
		biome.Depth = curr_biome.at("depth").to<float>();
		biome.Variation = curr_biome.at("variation").to<float>();
	};

	//start loading
	load(BR::Ocean, "ocean");
	load(BR::Plains, "plains");
	load(BR::Desert, "desert");
	load(BR::Forest, "forest");

	//finally register all biomes
	BR::registerBiomes();
}