#include "STPTerrainParaLoader.h"

//Biome Registry, just a demo program
#include "../World/Biomes/STPBiomeRegistry.h"
//System
#include <string>

using namespace STPDemo;
using namespace SuperTerrainPlus;

using std::string;
using std::pair;

using glm::uvec2;
using glm::vec2;
using glm::vec3;

STPEnvironment::STPMeshSetting STPTerrainParaLoader::getRenderingSetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPMeshSetting rendering_options;
	STPEnvironment::STPMeshSetting::STPTessellationSetting& tess_options = rendering_options.TessSetting;
	STPEnvironment::STPMeshSetting::STPTextureRegionSmoothSetting& smooth_options = rendering_options.RegionSmoothSetting;
	STPEnvironment::STPMeshSetting::STPTextureScaleDistanceSetting& scale_options = rendering_options.RegionScaleSetting;

	rendering_options.Strength = section("strength").to<float>();
	rendering_options.Altitude = section("altitude").to<float>();
	
	tess_options.MinTessLevel = section("minTess").to<float>();
	tess_options.MaxTessLevel = section("maxTess").to<float>();
	tess_options.FurthestTessDistance = section("furthestDistance").to<float>();
	tess_options.NearestTessDistance = section("nearestDistance").to<float>();

	smooth_options.KernelRadius = section("kernel_radius").to<unsigned int>();
	smooth_options.KernelScale = section("kernel_scale").to<float>();
	smooth_options.NoiseScale = section("noise_scale").to<unsigned int>();

	scale_options.PrimaryFar = section("uv_scale_primary_far").to<float>();
	scale_options.SecondaryFar = section("uv_scale_secondary_far").to<float>();
	scale_options.TertiaryFar = section("uv_scale_tertiary_far").to<float>();

	return rendering_options;
}

STPEnvironment::STPChunkSetting STPTerrainParaLoader::getChunkSetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPChunkSetting chunks_options;

	chunks_options.MapSize = uvec2(
		section("heightmap2DSizeX").to<unsigned int>(),
		section("heightmap2DSizeZ").to<unsigned int>()
	);
	chunks_options.ChunkSize = uvec2(
		section("chunkSizeX").to<unsigned int>(),
		section("chunkSizeZ").to<unsigned int>()
	);
	chunks_options.RenderedChunk = uvec2(
		section("renderedSizeX").to<unsigned int>(),
		section("renderedSizeZ").to<unsigned int>()
	);
	chunks_options.ChunkOffset = vec3(
		section("chunkOffsetX").to<float>(),
		section("chunkOffsetY").to<float>(),
		section("chunkOffsetZ").to<float>()
	);
	chunks_options.MapOffset = vec2(
		section("mapOffsetX").to<float>(),
		section("mapOffsetZ").to<float>()
	);
	chunks_options.FreeSlipChunk = uvec2(
		section("freeSlipX").to<unsigned int>(),
		section("freeSlipZ").to<unsigned int>()
	);
	chunks_options.ChunkScaling = section("chunkScale").to<float>();

	return chunks_options;
}

STPEnvironment::STPHeightfieldSetting STPTerrainParaLoader::getGeneratorSetting(const SIMPLE::SISection& section, glm::uvec2 slipRange) {
	//get the default settings
	STPEnvironment::STPHeightfieldSetting launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Seed = section("seed").to<STPDiversity::Seed>();
	launch_options.setErosionBrushRadius(slipRange, section("brush_radius").to<unsigned int>());
	launch_options.Inertia = section("inertia").to<float>();
	launch_options.SedimentCapacityFactor = section("sediment_capacity_factor").to<float>();
	launch_options.minSedimentCapacity = section("min_sediment_capacity").to<float>();
	launch_options.initWaterVolume = section("init_water_volume").to<float>();
	launch_options.minWaterVolume = section("min_water_volume").to<float>();
	launch_options.Friction = section("friction").to<float>();
	launch_options.initSpeed = section("init_speed").to<float>();
	launch_options.ErodeSpeed = section("erode_speed").to<float>();
	launch_options.DepositSpeed = section("deposit_speed").to<float>();
	launch_options.EvaporateSpeed = section("evaporate_speed").to<float>();
	launch_options.Gravity = section("gravity").to<float>();
	launch_options.RainDropCount = section("iteration").to<unsigned int>();

	//return the value
	return launch_options;
}

STPEnvironment::STPSimplexNoiseSetting STPTerrainParaLoader::getSimplexSetting(const SIMPLE::SISection& section) {
	auto noise_option = STPEnvironment::STPSimplexNoiseSetting();

	noise_option.Seed = section("seed").to<unsigned long long>();
	noise_option.Distribution = section("distribution").to<unsigned int>();
	noise_option.Offset = section("offset").to<double>();

	return noise_option;
}

pair<STPEnvironment::STPSunSetting, STPEnvironment::STPAtmosphereSetting> STPTerrainParaLoader::getSkySetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPSunSetting sun;
	sun.DayLength = section("day_length").to<size_t>();
	sun.DayStartOffset = section("day_start").to<size_t>();
	sun.YearLength = section("year_length").to<unsigned int>();
	sun.Obliquity = section("axial_tilt").to<double>();
	sun.Latitude = section("latitude").to<double>();

	STPEnvironment::STPAtmosphereSetting atmo;
	atmo.SunIntensity = section("sun_intensity").to<float>();
	atmo.PlanetRadius = section("planet_radius").to<float>();
	atmo.AtmosphereRadius = section("atmoshpere_radius").to<float>();
	atmo.ViewAltitude = section("view_altitude").to<float>();

	atmo.RayleighCoefficient = vec3(
		section("rayleigh_coefX").to<float>(),
		section("rayleigh_coefY").to<float>(),
		section("rayleigh_coefZ").to<float>()
	);
	atmo.MieCoefficient = section("mie_coef").to<float>();
	atmo.RayleighScale = section("rayleigh_scale").to<float>();
	atmo.MieScale = section("mie_scale").to<float>();
	atmo.MieScatteringDirection = section("mie_dir").to<float>();

	atmo.PrimaryRayStep = section("primary_step").to<unsigned int>();
	atmo.SecondaryRayStep = section("secondary_step").to<unsigned int>();

	return pair(sun, atmo);
}

STPEnvironment::STPOcclusionKernelSetting STPTerrainParaLoader::getAOSetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPOcclusionKernelSetting ao_kernel;

	ao_kernel.RandomSampleSeed = section("sample_seed").to<unsigned long long>();
	ao_kernel.RotationVectorSize = uvec2(
		section("rotation_vector_sizeX").to<unsigned int>(),
		section("rotation_vector_sizeY").to<unsigned int>()
	);
	ao_kernel.SampleRadius = section("sample_radius").to<float>();
	ao_kernel.Bias = section("bias").to<float>();

	return ao_kernel;
}

void STPTerrainParaLoader::loadBiomeParameters(const SIMPLE::SIStorage& biomeini) {
	using namespace STPDiversity;
	typedef STPDemo::STPBiomeRegistry BR;
	auto load = [&biomeini](STPDemo::STPBiome& biome, string name) -> void {
		STPDemo::STPBiomeSettings props;
		const SIMPLE::SISection& curr_biome = biomeini[name];

		//assigning props
		props.Name = curr_biome("name")();
		props.ID = curr_biome("id").to<Sample>();
		props.Temperature = curr_biome("temperature").to<float>();
		props.Precipitation = curr_biome("precipitation").to<float>();
		props.Scale = curr_biome("scale").to<float>();
		props.Octave = curr_biome("octave").to<unsigned int>();
		props.Persistence = curr_biome("persistence").to<float>();
		props.Lacunarity = curr_biome("lacunarity").to<float>();
		props.Depth = curr_biome("depth").to<float>();
		props.Variation = curr_biome("variation").to<float>();

		//modify biome settings
		biome.updateProperties(props);
	};

	//start loading
	load(BR::OCEAN, "ocean");
	load(BR::PLAINS, "plains");
	load(BR::DESERT, "desert");
	load(BR::FOREST, "forest");

	//finally register all biomes
	BR::registerBiomes();
}