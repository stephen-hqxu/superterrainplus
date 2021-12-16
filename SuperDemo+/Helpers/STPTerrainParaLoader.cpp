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

static constexpr char* RenderingVariables[] = {
	"strength",
	"altitude",
	"LoDfactor",
	"minTess",
	"maxTess",
	"furthestDistance",
	"nearestDistance"
};

static constexpr char* ChunkVariables[] = {
	"heightmap2DSizeX",
	"heightmap2DSizeZ",
	"chunkSizeX",
	"chunkSizeZ",
	"renderedSizeX",
	"renderedSizeZ",
	"chunkOffsetX",
	"chunkOffsetY",
	"chunkOffsetZ",
	"mapOffsetX",
	"mapOffsetZ",
	"freeSlipX",
	"freeSlipZ",
	"chunkScale"
};

static constexpr char* GeneratorVariables[] = {
	"seed",
	"brush_radius",
	"inertia",
	"sediment_capacity_factor",
	"min_sediment_capacity",
	"init_water_volume",
	"min_water_volume",
	"friction",
	"init_speed",
	"erode_speed",
	"deposit_speed",
	"evaporate_speed",
	"gravity",
	"iteration"
};

static constexpr char* SimplexVariables[] = {
	"seed",
	"distribution",
	"offset"
};

static constexpr char* BiomeVariables[] = {
	"name",
	"id",
	"temperature",
	"precipitation",
	"scale",
	"octave",
	"persistence",
	"lacunarity",
	"depth",
	"variation"
};

static constexpr char* SunVariables[] = {
	"day_length",
	"day_start",
	"year_length",
	"axial_tilt",
	"latitude",
	"sunset_angle",
	"sunrise_angle",
	"cycle_angle_offset"
};

static constexpr char* AtomshpereVariables[] = {
	"sun_intensity",
	"planet_radius",
	"atomshpere_radius",
	"view_altitude",
	"rayleigh_coefX",
	"rayleigh_coefY",
	"rayleigh_coefZ",
	"mie_coef",
	"rayleigh_scale",
	"mie_scale",
	"mie_dir",
	"primary_step",
	"secondary_step"
};

STPEnvironment::STPMeshSetting STPTerrainParaLoader::getRenderingSetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPMeshSetting rendering_options;
	STPEnvironment::STPMeshSetting::STPTessellationSetting tess_options;

	rendering_options.Strength = section(RenderingVariables[0]).to<float>();
	rendering_options.Altitude = section(RenderingVariables[1]).to<float>();
	rendering_options.LoDShiftFactor = section(RenderingVariables[2]).to<float>();
	
	tess_options.MinTessLevel = section(RenderingVariables[3]).to<float>();
	tess_options.MaxTessLevel = section(RenderingVariables[4]).to<float>();
	tess_options.FurthestTessDistance = section(RenderingVariables[5]).to<float>();
	tess_options.NearestTessDistance = section(RenderingVariables[6]).to<float>();

	rendering_options.TessSetting = tess_options;

	return rendering_options;
}

STPEnvironment::STPChunkSetting STPTerrainParaLoader::getChunkSetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPChunkSetting chunks_options;

	chunks_options.MapSize = uvec2(
		section(ChunkVariables[0]).to<unsigned int>(),
		section(ChunkVariables[1]).to<unsigned int>()
	);
	chunks_options.ChunkSize = uvec2(
		section(ChunkVariables[2]).to<unsigned int>(),
		section(ChunkVariables[3]).to<unsigned int>()
	);
	chunks_options.RenderedChunk = uvec2(
		section(ChunkVariables[4]).to<unsigned int>(),
		section(ChunkVariables[5]).to<unsigned int>()
	);
	chunks_options.ChunkOffset = vec3(
		section(ChunkVariables[6]).to<float>(),
		section(ChunkVariables[7]).to<float>(),
		section(ChunkVariables[8]).to<float>()
	);
	chunks_options.MapOffset = vec2(
		section(ChunkVariables[9]).to<float>(),
		section(ChunkVariables[10]).to<float>()
	);
	chunks_options.FreeSlipChunk = uvec2(
		section(ChunkVariables[11]).to<unsigned int>(),
		section(ChunkVariables[12]).to<unsigned int>()
	);
	chunks_options.ChunkScaling = section(ChunkVariables[13]).to<float>();

	return chunks_options;
}

STPEnvironment::STPHeightfieldSetting STPTerrainParaLoader::getGeneratorSetting(const SIMPLE::SISection& section, glm::uvec2 slipRange) {
	//get the default settings
	STPEnvironment::STPHeightfieldSetting launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Seed = section(GeneratorVariables[0]).to<STPDiversity::Seed>();
	launch_options.setErosionBrushRadius(slipRange, section(GeneratorVariables[1]).to<unsigned int>());
	launch_options.Inertia = section(GeneratorVariables[2]).to<float>();
	launch_options.SedimentCapacityFactor = section(GeneratorVariables[3]).to<float>();
	launch_options.minSedimentCapacity = section(GeneratorVariables[4]).to<float>();
	launch_options.initWaterVolume = section(GeneratorVariables[5]).to<float>();
	launch_options.minWaterVolume = section(GeneratorVariables[6]).to<float>();
	launch_options.Friction = section(GeneratorVariables[7]).to<float>();
	launch_options.initSpeed = section(GeneratorVariables[8]).to<float>();
	launch_options.ErodeSpeed = section(GeneratorVariables[9]).to<float>();
	launch_options.DepositSpeed = section(GeneratorVariables[10]).to<float>();
	launch_options.EvaporateSpeed = section(GeneratorVariables[11]).to<float>();
	launch_options.Gravity = section(GeneratorVariables[12]).to<float>();
	launch_options.RainDropCount = section(GeneratorVariables[13]).to<unsigned int>();

	//return the value
	return launch_options;
}

STPEnvironment::STPSimplexNoiseSetting STPTerrainParaLoader::getSimplexSetting(const SIMPLE::SISection& section) {
	auto noise_option = STPEnvironment::STPSimplexNoiseSetting();

	noise_option.Seed = section(SimplexVariables[0]).to<STPDiversity::Seed>();
	noise_option.Distribution = section(SimplexVariables[1]).to<unsigned int>();
	noise_option.Offset = section(SimplexVariables[2]).to<double>();

	return noise_option;
}

pair<STPEnvironment::STPSunSetting, STPEnvironment::STPAtomsphereSetting> STPTerrainParaLoader::getSkySetting(const SIMPLE::SISection& section) {
	STPEnvironment::STPSunSetting sun;
	sun.DayLength = section(SunVariables[0]).to<size_t>();
	sun.DayStartOffset = section(SunVariables[1]).to<size_t>();
	sun.YearLength = section(SunVariables[2]).to<unsigned int>();
	sun.Obliquity = section(SunVariables[3]).to<double>();
	sun.Latitude = section(SunVariables[4]).to<double>();
	sun.SunsetAngle = section(SunVariables[5]).to<double>();
	sun.SunriseAngle = section(SunVariables[6]).to<double>();
	sun.CycleAngleOffset = section(SunVariables[7]).to<double>();

	STPEnvironment::STPAtomsphereSetting atom;
	atom.SunIntensity = section(AtomshpereVariables[0]).to<float>();
	atom.PlanetRadius = section(AtomshpereVariables[1]).to<float>();
	atom.AtomsphereRadius = section(AtomshpereVariables[2]).to<float>();
	atom.ViewAltitude = section(AtomshpereVariables[3]).to<float>();

	atom.RayleighCoefficient = vec3(
		section(AtomshpereVariables[4]).to<float>(),
		section(AtomshpereVariables[5]).to<float>(),
		section(AtomshpereVariables[6]).to<float>()
	);
	atom.MieCoefficient = section(AtomshpereVariables[7]).to<float>();
	atom.RayleighScale = section(AtomshpereVariables[8]).to<float>();
	atom.MieScale = section(AtomshpereVariables[9]).to<float>();
	atom.MieScatteringDirection = section(AtomshpereVariables[10]).to<float>();

	atom.PrimaryRayStep = section(AtomshpereVariables[11]).to<unsigned int>();
	atom.SecondaryRayStep = section(AtomshpereVariables[12]).to<unsigned int>();

	return pair(sun, atom);
}

void STPTerrainParaLoader::loadBiomeParameters(const SIMPLE::SIStorage& biomeini) {
	using namespace STPDiversity;
	typedef STPDemo::STPBiomeRegistry BR;
	auto load = [&biomeini](STPDemo::STPBiome& biome, string name) -> void {
		STPDemo::STPBiomeSettings props;
		const SIMPLE::SISection& curr_biome = biomeini[name];

		//assigning props
		props.Name = curr_biome(BiomeVariables[0])();
		props.ID = curr_biome(BiomeVariables[1]).to<Sample>();
		props.Temperature = curr_biome(BiomeVariables[2]).to<float>();
		props.Precipitation = curr_biome(BiomeVariables[3]).to<float>();
		props.Scale = curr_biome(BiomeVariables[4]).to<float>();
		props.Octave = curr_biome(BiomeVariables[5]).to<unsigned int>();
		props.Persistence = curr_biome(BiomeVariables[6]).to<float>();
		props.Lacunarity = curr_biome(BiomeVariables[7]).to<float>();
		props.Depth = curr_biome(BiomeVariables[8]).to<float>();
		props.Variation = curr_biome(BiomeVariables[9]).to<float>();

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