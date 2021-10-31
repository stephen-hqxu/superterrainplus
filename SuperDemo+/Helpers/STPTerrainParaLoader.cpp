#include "STPTerrainParaLoader.h"

//Biome Registry, just a demo program
#include "../World/Biomes/STPBiomeRegistry.h"
//System
#include <string>

using namespace STPDemo;
using namespace SuperTerrainPlus;

using std::string;

using glm::uvec2;
using glm::vec2;
using glm::vec3;

static constexpr char* Procedural2DINFRenderingVariables[7] = {
	"strength",
	"altitude",
	"LoDfactor",
	"minTess",
	"maxTess",
	"furthestDistance",
	"nearestDistance"
};

static constexpr char* Procedural2DINFChunksVariables[14] = {
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

static constexpr char* Procedural2DINFGeneratorVariables[14] = {
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

static constexpr char* Simplex2DNoiseVariables[3] = {
	"seed",
	"distribution",
	"offset"
};

static constexpr char* BiomeVariables[10]{
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

STPEnvironment::STPMeshSetting STPTerrainParaLoader::getProcedural2DINFRenderingParameter(const SIMPLE::SISection& section) {
	STPEnvironment::STPMeshSetting rendering_options;
	STPEnvironment::STPMeshSetting::STPTessellationSetting tess_options;

	rendering_options.Strength = section(Procedural2DINFRenderingVariables[0]).to<float>();
	rendering_options.Altitude = section(Procedural2DINFRenderingVariables[1]).to<float>();
	rendering_options.LoDShiftFactor = section(Procedural2DINFRenderingVariables[2]).to<float>();
	
	tess_options.MinTessLevel = section(Procedural2DINFRenderingVariables[3]).to<float>();
	tess_options.MaxTessLevel = section(Procedural2DINFRenderingVariables[4]).to<float>();
	tess_options.FurthestTessDistance = section(Procedural2DINFRenderingVariables[5]).to<float>();
	tess_options.NearestTessDistance = section(Procedural2DINFRenderingVariables[6]).to<float>();

	rendering_options.TessSetting = tess_options;

	return rendering_options;
}

STPEnvironment::STPChunkSetting STPTerrainParaLoader::getProcedural2DINFChunksParameter(const SIMPLE::SISection& section) {
	STPEnvironment::STPChunkSetting chunks_options;

	chunks_options.MapSize = uvec2(
		section(Procedural2DINFChunksVariables[0]).to<unsigned int>(),
		section(Procedural2DINFChunksVariables[1]).to<unsigned int>()
	);
	chunks_options.ChunkSize = uvec2(
		section(Procedural2DINFChunksVariables[2]).to<unsigned int>(),
		section(Procedural2DINFChunksVariables[3]).to<unsigned int>()
	);
	chunks_options.RenderedChunk = uvec2(
		section(Procedural2DINFChunksVariables[4]).to<unsigned int>(),
		section(Procedural2DINFChunksVariables[5]).to<unsigned int>()
	);
	chunks_options.ChunkOffset = vec3(
		section(Procedural2DINFChunksVariables[6]).to<float>(),
		section(Procedural2DINFChunksVariables[7]).to<float>(),
		section(Procedural2DINFChunksVariables[8]).to<float>()
	);
	chunks_options.MapOffset = vec2(
		section(Procedural2DINFChunksVariables[9]).to<float>(),
		section(Procedural2DINFChunksVariables[10]).to<float>()
	);
	chunks_options.FreeSlipChunk = uvec2(
		section(Procedural2DINFChunksVariables[11]).to<unsigned int>(),
		section(Procedural2DINFChunksVariables[12]).to<unsigned int>()
	);
	chunks_options.ChunkScaling = section(Procedural2DINFChunksVariables[13]).to<float>();

	return chunks_options;
}

STPEnvironment::STPHeightfieldSetting STPTerrainParaLoader::getProcedural2DINFGeneratorParameter(const SIMPLE::SISection& section, glm::uvec2 slipRange) {
	//get the default settings
	STPEnvironment::STPHeightfieldSetting launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Seed = section(Procedural2DINFGeneratorVariables[0]).to<STPDiversity::Seed>();
	launch_options.setErosionBrushRadius(slipRange, section(Procedural2DINFGeneratorVariables[1]).to<unsigned int>());
	launch_options.Inertia = section(Procedural2DINFGeneratorVariables[2]).to<float>();
	launch_options.SedimentCapacityFactor = section(Procedural2DINFGeneratorVariables[3]).to<float>();
	launch_options.minSedimentCapacity = section(Procedural2DINFGeneratorVariables[4]).to<float>();
	launch_options.initWaterVolume = section(Procedural2DINFGeneratorVariables[5]).to<float>();
	launch_options.minWaterVolume = section(Procedural2DINFGeneratorVariables[6]).to<float>();
	launch_options.Friction = section(Procedural2DINFGeneratorVariables[7]).to<float>();
	launch_options.initSpeed = section(Procedural2DINFGeneratorVariables[8]).to<float>();
	launch_options.ErodeSpeed = section(Procedural2DINFGeneratorVariables[9]).to<float>();
	launch_options.DepositSpeed = section(Procedural2DINFGeneratorVariables[10]).to<float>();
	launch_options.EvaporateSpeed = section(Procedural2DINFGeneratorVariables[11]).to<float>();
	launch_options.Gravity = section(Procedural2DINFGeneratorVariables[12]).to<float>();
	launch_options.RainDropCount = section(Procedural2DINFGeneratorVariables[13]).to<unsigned int>();

	//return the value
	return launch_options;
}

STPEnvironment::STPSimplexNoiseSetting STPTerrainParaLoader::getSimplex2DNoiseParameter(const SIMPLE::SISection& section) {
	auto noise_option = STPEnvironment::STPSimplexNoiseSetting();

	noise_option.Seed = section(Simplex2DNoiseVariables[0]).to<STPDiversity::Seed>();
	noise_option.Distribution = section(Simplex2DNoiseVariables[1]).to<unsigned int>();
	noise_option.Offset = section(Simplex2DNoiseVariables[2]).to<double>();

	return noise_option;
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