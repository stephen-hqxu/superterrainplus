#include "STPWorldManager.h"

//Error
#include <SuperTerrain+/Exception/STPInvalidArgument.h>

//System
#include <array>
#include <algorithm>
#include <execution>
#include <string_view>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

//Texture Loader
#include "../Helpers/STPTextureStorage.h"
//Texture Splatting
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>
#include <SuperAlgorithm+/Parser/STPTextureDefinitionLanguage.h>

//GLAD
#include <glad/glad.h>

using namespace STPDemo;
using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;
using STPAlgorithm::STPTextureDefinitionLanguage;

using std::invalid_argument;
using std::make_optional;
using std::make_unique;

using std::array;
using std::string;
using std::string_view;
using std::pair;
using std::make_pair;
using std::for_each;
using std::generate;

using glm::ivec2;
using glm::uvec2;

class STPWorldManager::STPWorldSplattingAgent {
private:

	constexpr static size_t TextureCount = 16u;
	//All loaded texture data
	STPTextureStorage LoadedData[STPWorldSplattingAgent::TextureCount];
	//List of all texture name
	constexpr static const char* Filename[] = {
		"darkrock_color.jpg",
		"darkrock_normal.jpg",
		"darkrock_ao.jpg",
		"grass_color.jpg",
		"grass_normal.jpg",
		"grass_ao.jpg",
		"mossrock_color.jpg",
		"mossrock_normal.jpg",
		"mossrock_ao.jpg",
		"redrock_color.jpg",
		"redrock_normal.jpg",
		"redrock_ao.jpg",
		"sand_color.jpg",
		"sand_normal.jpg",
		"sand_ao.jpg",
		"soil_color.jpg"
	};
	constexpr static char TDLFilename[] = "./Script/STPBiomeSplatRule.tdl";

	//Group ID recording
	STPTextureInformation::STPMapGroupID x1024_srgb, x1024_rgb, x1024_r;

	/**
	 * @brief Determine the texture type based on the filename.
	 * @param filename The filename of the texture.
	 * @return The type of texture.
	*/
	static STPTextureType getType(const string_view& filename) {
		//find the string representation of the type
		const size_t type_start = filename.find('_') + 1u,
			type_end = filename.find('.');
		const string_view typeStr = filename.substr(type_start, type_end - type_start);

		//convert string to texture type
		if (typeStr == "color") {
			return STPTextureType::Albedo;
		}
		if (typeStr == "normal") {
			return STPTextureType::Normal;
		}
		if (typeStr == "ao") {
			return STPTextureType::AmbientOcclusion;
		}

		throw STPException::STPInvalidArgument("Cannot determine the type of this texture");
	}

public:

	//A texture database preloaded with configurations.
	STPDiversity::STPTextureDatabase Database;

	/**
	 * @brief Init STPWorldSplattingAgent.
	 * @param prefix The filename prefix for all texture filenames.
	*/
	STPWorldSplattingAgent(const string& prefix) {
		array<pair<unsigned int, const char*>, STPWorldSplattingAgent::TextureCount> IndexedFilename;
		//load up indices
		generate(IndexedFilename.begin(), IndexedFilename.end(), [index = 0u]() mutable {
			const unsigned int i = index++;
			return make_pair(i, STPWorldSplattingAgent::Filename[i]);
		});

		//load all texture from the file system
		for_each(std::execution::par, IndexedFilename.cbegin(), IndexedFilename.cend(),
			[&texArr = this->LoadedData, &prefix](const auto& data) {
			const auto [index, filename] = data;
			//ambient occlusion texture only has one channel
			const int channel = STPWorldSplattingAgent::getType(filename) == STPTextureType::AmbientOcclusion ? 1 : 3;

			//all our images don't have alpha channel.
			//loading texture from file system can be done in parallel
			//insertion into the container needs to be safe.
			texArr[index] = STPTextureStorage(prefix + '/' + filename, channel);
		});

		//create group
		STPTextureDatabase::STPMapGroupDescription tex_desc = { };
		tex_desc.Dimension = uvec2(1024u);
		tex_desc.MipMapLevel = 7u;
		tex_desc.PixelFormat = GL_UNSIGNED_BYTE;
		tex_desc.ChannelFormat = GL_RGB;
		tex_desc.InteralFormat = GL_SRGB8;
		this->x1024_srgb = this->Database.addMapGroup(tex_desc);
		tex_desc.InteralFormat = GL_RGB8;
		this->x1024_rgb = this->Database.addMapGroup(tex_desc);
		tex_desc.ChannelFormat = GL_RED;
		tex_desc.InteralFormat = GL_R8;
		this->x1024_r = this->Database.addMapGroup(tex_desc);

		const string rawTDL = STPFile::read(STPWorldSplattingAgent::TDLFilename);
		const STPTextureDefinitionLanguage TDLParser(rawTDL);
		//build texture splatting rules
		const STPTextureDefinitionLanguage::STPTextureVariable textureName = TDLParser(this->Database);
		//build database with texture data
		for (unsigned int i = 0u; i < STPWorldSplattingAgent::TextureCount; i++) {
			//grab the texture ID using the texture name
			const string_view currTexFile(STPWorldSplattingAgent::Filename[i]);
			//our filename always follows this pattern: (texture name)_(type).(suffix), we can search using that
			const STPTextureInformation::STPTextureID currTexID = textureName.at(currTexFile.substr(0u, currTexFile.find_first_of('_'))).first;

			STPTextureType texType = STPWorldSplattingAgent::getType(currTexFile);
			STPTextureInformation::STPMapGroupID texGroup = 0u;
			//change group ID conditionally
			switch (texType) {
			case STPTextureType::Albedo:
				//colour texture is usually in gamma space, we need to transform it to linear space.
				texGroup = this->x1024_srgb;
				break;
			case STPTextureType::AmbientOcclusion:
				texGroup = this->x1024_r;
				break;
			default:
				texGroup = this->x1024_rgb;
				break;
			}
			this->Database.addMap(currTexID, texType, texGroup, this->LoadedData[i].texture());
		}
	}

	STPWorldSplattingAgent(const STPWorldSplattingAgent&) = delete;

	STPWorldSplattingAgent(STPWorldSplattingAgent&&) = delete;

	STPWorldSplattingAgent& operator=(const STPWorldSplattingAgent&) = delete;

	STPWorldSplattingAgent& operator=(STPWorldSplattingAgent&&) = delete;

	~STPWorldSplattingAgent() = default;

};

STPWorldManager::STPWorldManager(const string& tex_filename_prefix, const STPEnvironment::STPChunkSetting& chunk_setting,
	const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) :
	SharedProgram(chunk_setting, simplex_setting), linkStatus(false),
	Texture(make_unique<STPWorldManager::STPWorldSplattingAgent>(tex_filename_prefix)) {

}

STPWorldManager::~STPWorldManager() = default;

void STPWorldManager::linkProgram(const STPEnvironment::STPChunkSetting& chunk_setting,
	const STPEnvironment::STPHeightfieldSetting& heightfield_setting) {
	this->linkStatus = false;
	//error checking
	if (!this->BiomeFactory) {
		throw invalid_argument("Biome factory not attached.");
	}
	if (!this->TextureFactory) {
		throw invalid_argument("Texture factory not attached.");
	}

	//create generator and storage unit first
	STPHeightfieldGenerator::STPGeneratorSetup setup = { };
	setup.ChunkSetting = &chunk_setting;
	setup.HeightfieldSetting = &heightfield_setting;
	setup.DiversityGenerator = this->DiversityGenerator.get();
	//TODO: correctly calculate the thread occupancy
	setup.ConcurrencyLevelHint = 5u;
	this->ChunkGenerator.emplace(setup);

	//create the world pipeline
	STPWorldPipeline::STPPipelineSetup pipeStage = { };
	pipeStage.BiomemapGenerator = this->BiomeFactory.get();
	pipeStage.HeightfieldGenerator = &(*this->ChunkGenerator);
	pipeStage.SplatmapGenerator = this->TextureFactory.get();
	pipeStage.ChunkSetting = &chunk_setting;
	this->Pipeline.emplace(pipeStage);
	
	this->linkStatus = true;
}

STPWorldManager::operator bool() const {
	return this->linkStatus;
}

SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPDatabaseView STPWorldManager::getTextureDatabase() const {
	return this->Texture->Database.visit();
}

STPWorldPipeline& STPWorldManager::getPipeline() {
	return *this->Pipeline;
}