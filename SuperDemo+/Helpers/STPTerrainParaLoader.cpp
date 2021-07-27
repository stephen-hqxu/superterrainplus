#include "STPTerrainParaLoader.h"

//Biome Registry, just a demo program
#include "../World/Biomes/STPBiomeRegistry.h"

using namespace SuperTerrainPlus;

using std::string;
using std::stof;
using std::stoi;
using std::stod;

using glm::uvec2;
using glm::vec2;
using glm::vec3;

const string STPTerrainParaLoader::Procedural2DINFRenderingVariables[6] = {
	"altitude",
	"LoDfactor",
	"minTess",
	"maxTess",
	"furthestDistance",
	"nearestDistance"
};

const string STPTerrainParaLoader::Procedural2DINFChunksVariables[14] = {
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

const string STPTerrainParaLoader::Procedural2DINFGeneratorVariables[15] = {
	"seed",
	"strength",
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

const string STPTerrainParaLoader::Simplex2DNoiseVariables[3] = {
	"seed",
	"distribution",
	"offset"
};

const string STPTerrainParaLoader::BiomeVariables[10]{
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

STPSettings::STPMeshSettings STPTerrainParaLoader::getProcedural2DINFRenderingParameters(const SIMPLE::SISection& section) {
	STPSettings::STPMeshSettings rendering_options;
	STPSettings::STPMeshSettings::STPTessellationSettings tess_options;

	rendering_options.Altitude = stof(section(STPTerrainParaLoader::Procedural2DINFRenderingVariables[0]));
	rendering_options.LoDShiftFactor = stof(section(STPTerrainParaLoader::Procedural2DINFRenderingVariables[1]));
	
	tess_options.MinTessLevel = stof(section(STPTerrainParaLoader::Procedural2DINFRenderingVariables[2]));
	tess_options.MaxTessLevel = stof(section(STPTerrainParaLoader::Procedural2DINFRenderingVariables[3]));
	tess_options.FurthestTessDistance = stof(section(STPTerrainParaLoader::Procedural2DINFRenderingVariables[4]));
	tess_options.NearestTessDistance = stof(section(STPTerrainParaLoader::Procedural2DINFRenderingVariables[5]));

	rendering_options.TessSettings = tess_options;

	return rendering_options;
}

STPSettings::STPChunkSettings STPTerrainParaLoader::getProcedural2DINFChunksParameters(const SIMPLE::SISection& section) {
	STPSettings::STPChunkSettings chunks_options;

	chunks_options.MapSize = uvec2(
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[0])),
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[1]))
	);
	chunks_options.ChunkSize = uvec2(
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[2])),
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[3]))
	);
	chunks_options.RenderedChunk = uvec2(
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[4])),
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[5]))
	);
	chunks_options.ChunkOffset = vec3(
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[6])),
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[7])),
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[8]))
	);
	chunks_options.MapOffset = vec2(
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[9])),
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[10]))
	);
	chunks_options.FreeSlipChunk = uvec2(
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[11])),
		stoul(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[12]))
	);
	chunks_options.ChunkScaling = stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[13]));

	return chunks_options;
}

STPSettings::STPHeightfieldSettings STPTerrainParaLoader::getProcedural2DINFGeneratorParameters(const SIMPLE::SISection& section, glm::uvec2 slipRange) {
	//get the default settings
	STPSettings::STPHeightfieldSettings launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Seed = stoull(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[0]));
	launch_options.Strength = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[1]));
	launch_options.setErosionBrushRadius(make_uint2(slipRange.x, slipRange.y), stoul(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[2])));
	launch_options.Inertia = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[3]));
	launch_options.SedimentCapacityFactor = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[4]));
	launch_options.minSedimentCapacity = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[5]));
	launch_options.initWaterVolume = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[6]));
	launch_options.minWaterVolume = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[7]));
	launch_options.Friction = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[8]));
	launch_options.initSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[9]));
	launch_options.ErodeSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[10]));
	launch_options.DepositSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[11]));
	launch_options.EvaporateSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[12]));
	launch_options.Gravity = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[13]));
	launch_options.RainDropCount = stoul(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[14]));

	//return the value
	return launch_options;
}

STPSettings::STPSimplexNoiseSettings STPTerrainParaLoader::getSimplex2DNoiseParameters(const SIMPLE::SISection& section) {
	auto noise_option = STPSettings::STPSimplexNoiseSettings();

	noise_option.Seed = stoull(section(STPTerrainParaLoader::Simplex2DNoiseVariables[0]));
	noise_option.Distribution = stoul(section(STPTerrainParaLoader::Simplex2DNoiseVariables[1]));
	noise_option.Offset = stod(section(STPTerrainParaLoader::Simplex2DNoiseVariables[2]));

	return noise_option;
}

void STPTerrainParaLoader::loadBiomeParameters(const SIMPLE::SIStorage& biomeini) {
	using namespace STPDiversity;
	typedef STPDemo::STPBiomeRegistry BR;
	auto load = [&biomeini](STPDemo::STPBiome& biome, string name) -> void {
		STPDemo::STPBiomeSettings props;
		const SIMPLE::SISection& curr_biome = biomeini[name];

		//assigning props
		props.Name = curr_biome(STPTerrainParaLoader::BiomeVariables[0]);
		props.ID = static_cast<STPDiversity::Sample>(stoul(curr_biome(STPTerrainParaLoader::BiomeVariables[1])));
		props.Temperature = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[2]));
		props.Precipitation = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[3]));
		props.Scale = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[4]));
		props.Octave = static_cast<unsigned int>(stoull(curr_biome(STPTerrainParaLoader::BiomeVariables[5])));
		props.Persistence = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[6]));
		props.Lacunarity = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[7]));
		props.Depth = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[8]));
		props.Variation = stof(curr_biome(STPTerrainParaLoader::BiomeVariables[9]));

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