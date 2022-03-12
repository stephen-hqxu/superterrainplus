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

STPEnvironment::STPChunkSetting STPTerrainParaLoader::getChunkSetting(const SIMPLE::SISection& section) {
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
	chunks_options.ChunkOffset = vec3(
		section.at("chunkOffsetX").to<float>(),
		section.at("chunkOffsetY").to<float>(),
		section.at("chunkOffsetZ").to<float>()
	);
	chunks_options.MapOffset = vec2(
		section.at("mapOffsetX").to<float>(),
		section.at("mapOffsetZ").to<float>()
	);
	chunks_options.FreeSlipChunk = uvec2(
		section.at("freeSlipX").to<unsigned int>(),
		section.at("freeSlipZ").to<unsigned int>()
	);
	chunks_options.ChunkScaling = section.at("chunkScale").to<float>();

	return chunks_options;
}

STPEnvironment::STPHeightfieldSetting STPTerrainParaLoader::getGeneratorSetting(const SIMPLE::SISection& section, glm::uvec2 slipRange) {
	//get the default settings
	STPEnvironment::STPHeightfieldSetting launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Seed = section.at("seed").to<STPDiversity::Seed>();
	launch_options.setErosionBrushRadius(slipRange, section.at("brush_radius").to<unsigned int>());
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
	launch_options.RainDropCount = section.at("iteration").to<unsigned int>();

	//return the value
	return launch_options;
}

STPEnvironment::STPSimplexNoiseSetting STPTerrainParaLoader::getSimplexSetting(const SIMPLE::SISection& section) {
	auto noise_option = STPEnvironment::STPSimplexNoiseSetting();

	noise_option.Seed = section.at("seed").to<unsigned long long>();
	noise_option.Distribution = section.at("distribution").to<unsigned int>();
	noise_option.Offset = section.at("offset").to<double>();

	return noise_option;
}

pair<STPEnvironment::STPSunSetting, STPEnvironment::STPAtmosphereSetting> STPTerrainParaLoader::getSkySetting(const SIMPLE::SISection& section) {
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

STPEnvironment::STPOcclusionKernelSetting STPTerrainParaLoader::getAOSetting(const SIMPLE::SISection& section) {
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

void STPTerrainParaLoader::loadBiomeParameters(const SIMPLE::SIStorage& biomeini) {
	using namespace STPDiversity;
	typedef STPDemo::STPBiomeRegistry BR;
	auto load = [&biomeini](STPDemo::STPBiome& biome, string name) -> void {
		STPDemo::STPBiomeSettings props;
		const SIMPLE::SISection& curr_biome = biomeini.at(name);

		//assigning props
		props.Name = *curr_biome.at("name");
		props.ID = curr_biome.at("id").to<Sample>();
		props.Temperature = curr_biome.at("temperature").to<float>();
		props.Precipitation = curr_biome.at("precipitation").to<float>();
		props.Scale = curr_biome.at("scale").to<float>();
		props.Octave = curr_biome.at("octave").to<unsigned int>();
		props.Persistence = curr_biome.at("persistence").to<float>();
		props.Lacunarity = curr_biome.at("lacunarity").to<float>();
		props.Depth = curr_biome.at("depth").to<float>();
		props.Variation = curr_biome.at("variation").to<float>();

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