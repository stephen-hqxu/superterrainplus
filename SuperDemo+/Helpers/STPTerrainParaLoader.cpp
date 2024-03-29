#include "STPTerrainParaLoader.h"

//Biome Registry, just a demo program
#include "../World/Biomes/STPBiomeRegistry.h"
//System
#include <string_view>

namespace Env = SuperTerrainPlus::STPEnvironment;

using namespace STPDemo;

using SuperTerrainPlus::STPSeed_t;

using SuperTerrainPlus::STPAlgorithm::STPINIData::STPINIStorageView;
using SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView;

using std::string_view;
using std::pair;

using glm::uvec2;
using glm::dvec2;
using glm::vec2;
using glm::vec3;
using glm::dvec3;

Env::STPMeshSetting STPTerrainParaLoader::getRenderingSetting(const STPINISectionView& section) {
	Env::STPMeshSetting rendering_options = { };
	Env::STPTessellationSetting& tess_options = rendering_options.TessSetting;
	Env::STPMeshSetting::STPTextureRegionSmoothSetting& smooth_options = rendering_options.RegionSmoothSetting;
	Env::STPMeshSetting::STPTextureScaleDistanceSetting& scale_options = rendering_options.RegionScaleSetting;

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

Env::STPChunkSetting STPTerrainParaLoader::getChunkSetting(const STPINISectionView& section) {
	Env::STPChunkSetting chunks_options = { };

	chunks_options.MapSize = uvec2(
		section.at("heightmap2DSizeX").to<unsigned int>(),
		section.at("heightmap2DSizeZ").to<unsigned int>()
	);
	chunks_options.ChunkSize = uvec2(
		section.at("chunkSizeX").to<unsigned int>(),
		section.at("chunkSizeZ").to<unsigned int>()
	);
	chunks_options.SplatNearestNeighbour = uvec2(
		section.at("splatSizeX").to<unsigned int>(),
		section.at("splatSizeZ").to<unsigned int>()
	);
	chunks_options.RenderDistance = uvec2(
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
	chunks_options.DiversityNearestNeighbour = uvec2(
		section.at("diversityNeighbourX").to<unsigned int>(),
		section.at("diversityNeighbourY").to<unsigned int>()
	);
	chunks_options.ErosionNearestNeighbour = uvec2(
		section.at("erosionNeighbourX").to<unsigned int>(),
		section.at("erosionNeighbourY").to<unsigned int>()
	);
	chunks_options.ChunkScale = dvec2(
		section.at("chunkScaleX").to<double>(),
		section.at("chunkScaleY").to<double>()
	);

	return chunks_options;
}

Env::STPRainDropSetting STPTerrainParaLoader::getRainDropSetting(const STPINISectionView& section, const STPSeed_t generator_seed) {
	//get the default settings
	Env::STPRainDropSetting raindrop_options = { };
	
	raindrop_options.Seed = generator_seed;
	raindrop_options.RainDropCount = section.at("iteration").to<unsigned int>();

	raindrop_options.Inertia = section.at("inertia").to<float>();
	raindrop_options.SedimentCapacityFactor = section.at("sediment_capacity_factor").to<float>();
	raindrop_options.minSedimentCapacity = section.at("min_sediment_capacity").to<float>();
	raindrop_options.initWaterVolume = section.at("init_water_volume").to<float>();
	raindrop_options.minWaterVolume = section.at("min_water_volume").to<float>();
	raindrop_options.Friction = section.at("friction").to<float>();
	raindrop_options.initSpeed = section.at("init_speed").to<float>();
	raindrop_options.ErodeSpeed = section.at("erode_speed").to<float>();
	raindrop_options.DepositSpeed = section.at("deposit_speed").to<float>();
	raindrop_options.EvaporateSpeed = section.at("evaporate_speed").to<float>();
	raindrop_options.Gravity = section.at("gravity").to<float>();
	raindrop_options.ErosionBrushRadius = section.at("brush_radius").to<unsigned int>();

	//return the value
	return raindrop_options;
}

Env::STPSimplexNoiseSetting STPTerrainParaLoader::getSimplexSetting(const STPINISectionView& section, const STPSeed_t simplex_seed) {
	Env::STPSimplexNoiseSetting noise_option = { };

	noise_option.Seed = simplex_seed;
	noise_option.Distribution = section.at("distribution").to<unsigned int>();
	noise_option.Offset = section.at("offset").to<double>();

	return noise_option;
}

pair<Env::STPSunSetting, Env::STPAtmosphereSetting> STPTerrainParaLoader::getSkySetting(const STPINISectionView& section) {
	Env::STPSunSetting sun = { };
	sun.DayLength = section.at("day_length").to<unsigned int>();
	sun.DayStart = section.at("day_start").to<unsigned int>();
	sun.YearLength = section.at("year_length").to<unsigned int>();
	sun.YearStart = section.at("year_start").to<unsigned int>();
	sun.Obliquity = section.at("axial_tilt").to<double>();
	sun.Latitude = section.at("latitude").to<double>();

	Env::STPAtmosphereSetting atmo = { };
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

Env::STPStarfieldSetting STPTerrainParaLoader::getStarfieldSetting(const STPINISectionView& section, const STPSeed_t star_seed) {
	Env::STPStarfieldSetting star = { };

	star.Seed = star_seed;
	star.InitialLikelihood = section.at("init_likelihood").to<float>();
	star.OctaveLikelihoodMultiplier = section.at("likelihood_mul").to<float>();
	star.InitialScale = section.at("init_scale").to<float>();
	star.OctaveScaleMultiplier = section.at("scale_mul").to<float>();
	star.EdgeDistanceFalloff = section.at("edge_falloff").to<float>();
	star.ShineSpeed = section.at("shine_speed").to<float>();
	star.LuminosityMultiplier = section.at("brightness").to<float>();
	star.MinimumAltitude = section.at("min_altitude").to<float>();
	star.Octave = section.at("octave").to<unsigned int>();

	return star;
}

Env::STPAuroraSetting STPTerrainParaLoader::getAuroraSetting(const STPINISectionView& section) {
	Env::STPAuroraSetting aurora = { };
	auto& tri = aurora.Noise;
	auto& main_fractal = tri.MainNoise, &distortion_fractal = tri.DistortionNoise;

	main_fractal.InitialAmplitude = section.at("main_init_amplitude").to<float>();
	main_fractal.Persistence = section.at("main_persistence").to<float>();
	main_fractal.Lacunarity = section.at("main_lacunarity").to<float>();
	distortion_fractal.InitialAmplitude = section.at("distortion_init_amplitude").to<float>();
	distortion_fractal.Persistence = section.at("distortion_persistence").to<float>();
	distortion_fractal.Lacunarity = section.at("distortion_lacunarity").to<float>();

	tri.InitialDistortionFrequency = section.at("init_distortion_freq").to<float>();
	tri.Curvature = section.at("curvature").to<float>();
	tri.OctaveRotation = section.at("octave_rotation").to<float>();
	tri.AnimationSpeed = section.at("aurora_speed").to<float>();
	tri.Contrast = section.at("contrast").to<float>();
	tri.MaximumIntensity = section.at("max_intensity").to<float>();
	tri.Octave = section.at("tri_noise_octave").to<unsigned int>();

	aurora.AuroraSphereFlatness = section.at("flatness").to<float>();
	aurora.AuroraPlaneProjectionBias = section.at("projection_bias").to<float>();
	aurora.StepSize = section.at("step_size").to<float>();
	aurora.AltitudeFadeStart = section.at("fade_start").to<float>();
	aurora.AltitudeFadeEnd = section.at("fade_end").to<float>();
	aurora.LuminosityMultiplier = section.at("aurora_brightness").to<float>();
	aurora.Iteration = section.at("aurora_iteration").to<unsigned int>();

	return aurora;
}

Env::STPOcclusionKernelSetting STPTerrainParaLoader::getAOSetting(const STPINISectionView& section, const STPSeed_t ao_seed) {
	Env::STPOcclusionKernelSetting ao_kernel = { };

	ao_kernel.RandomSampleSeed = ao_seed;
	ao_kernel.RotationVectorSize = uvec2(
		section.at("rotation_vector_sizeX").to<unsigned int>(),
		section.at("rotation_vector_sizeY").to<unsigned int>()
	);
	ao_kernel.SampleRadius = section.at("sample_radius").to<float>();
	ao_kernel.Bias = section.at("bias").to<float>();

	return ao_kernel;
}

Env::STPWaterSetting STPTerrainParaLoader::getWaterSetting(const STPINISectionView& section, const float altitude) {
	Env::STPWaterSetting water = { };
	Env::STPTessellationSetting& water_tess = water.WaterMeshTess;
	Env::STPWaterSetting::STPWaterWaveSetting& water_wave = water.WaterWave;

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

Env::STPBidirectionalScatteringSetting STPTerrainParaLoader::getBSDFSetting(const STPINISectionView& section) {
	Env::STPBidirectionalScatteringSetting bsdf = { };

	bsdf.MaxRayDistance = section.at("max_ray_distance").to<float>();
	bsdf.DepthBias = section.at("depth_bias").to<float>();
	bsdf.RayResolution = section.at("ray_resolution").to<unsigned int>();
	bsdf.RayStep = section.at("ray_step").to<unsigned int>();

	return bsdf;
}

void STPTerrainParaLoader::loadBiomeParameters(const STPINIStorageView& biomeini) {
	namespace BR = STPDemo::STPBiomeRegistry;
	const auto load = [&biomeini](STPDemo::STPBiome& biome, const string_view name) -> void {
		const STPINISectionView& curr_biome = biomeini.at(name);

		//assigning props
		biome.Name = curr_biome.at("name").String;
		biome.ID = curr_biome.at("id").to<SuperTerrainPlus::STPSample_t>();
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