#include "STPTerrainParaLoader.h"

using namespace SuperTerrainPlus;

using std::stof;
using std::stoi;
using std::stod;

STPSettings::STPMeshSettings STPTerrainParaLoader::getProcedural2DINFRenderingParameters(SIMPLE::SISection& section) {
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

STPSettings::STPChunkSettings STPTerrainParaLoader::getProcedural2DINFChunksParameters(SIMPLE::SISection& section) {
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
	chunks_options.MapOffset = vec3(
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[9])),
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[10])),
		stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[11]))
	);
	chunks_options.ChunkScaling = stof(section(STPTerrainParaLoader::Procedural2DINFChunksVariables[12]));

	return chunks_options;
}

STPSettings::STPHeightfieldSettings STPTerrainParaLoader::getProcedural2DINFGeneratorParameters(SIMPLE::SISection& section, glm::uvec2 mapSize) {
	//get the default settings
	STPSettings::STPHeightfieldSettings launch_options;
	
	//set the parameter one by one, enjoy :)
	launch_options.Scale = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[0]));
	launch_options.Octave = stoul(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[1]));
	launch_options.Persistence = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[2]));
	launch_options.Lacunarity = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[3]));
	launch_options.Strength = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[4]));
	launch_options.setErosionBrushRadius(make_uint2(mapSize.x, mapSize.y), stoul(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[5])));
	launch_options.Inertia = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[6]));
	launch_options.SedimentCapacityFactor = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[7]));
	launch_options.minSedimentCapacity = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[8]));
	launch_options.initWaterVolume = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[9]));
	launch_options.minWaterVolume = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[10]));
	launch_options.Friction = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[11]));
	launch_options.initSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[12]));
	launch_options.ErodeSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[13]));
	launch_options.DepositSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[14]));
	launch_options.EvaporateSpeed = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[15]));
	launch_options.Gravity = stof(section(STPTerrainParaLoader::Procedural2DINFGeneratorVariables[16]));

	//return the value
	return launch_options;
}

STPSettings::STPSimplexNoiseSettings STPTerrainParaLoader::getSimplex2DNoiseParameters(SIMPLE::SISection& section, glm::uvec2 mapSize) {
	auto noise_option = STPSettings::STPSimplexNoiseSettings();

	noise_option.Seed = stoull(section(STPTerrainParaLoader::Simplex2DNoiseVariables[0]));
	noise_option.Distribution = stoul(section(STPTerrainParaLoader::Simplex2DNoiseVariables[1]));
	noise_option.Offset = stod(section(STPTerrainParaLoader::Simplex2DNoiseVariables[2]));
	noise_option.Dimension = make_uint2(mapSize.x, mapSize.y);

	return noise_option;
}