#include "STPWorldManager.h"

//System
#include <exception>
#include <array>
#include <algorithm>
#include <execution>
#include <string_view>

#include <fstream>
#include <sstream>

//Texture Loader
#include "../Helpers/STPTextureStorage.h"
//Texture Splatting
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDefinitionLanguage.h>

using namespace STPDemo;
using namespace SuperTerrainPlus;

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

	constexpr static size_t TextureCount = 6ull;
	//All loaded texture data
	STPTextureStorage LoadedData[STPWorldSplattingAgent::TextureCount];
	//List of all texture name
	constexpr static char* Filename[] = {
		"darkrock_color.jpg",
		"grass_color.jpg",
		"mossrock_color.jpg",
		"redrock_color.jpg",
		"sand_color.jpg",
		"soil_color.jpg"
	};

	//Splatting
	STPDiversity::STPTextureDefinitionLanguage TDLParser;

	constexpr static char TDLFilename[] = "./Script/STPBiomeSplatRule.tdl";

	/**
	 * @brief Read a TDL source codes from the local file system.
	 * @return The string representation of the source code.
	*/
	static string readTDL() {
		std::ifstream tdlFile(STPWorldSplattingAgent::TDLFilename);
		if (!tdlFile) {
			std::terminate();
		}

		//read all lines
		std::stringstream buffer;
		buffer << tdlFile.rdbuf();

		return buffer.str();
	}

public:

	//A texture database preloaded with configurations.
	STPDiversity::STPTextureDatabase Database;

	/**
	 * @brief Init STPWorldSplattingAgent.
	 * @param prefix The filename prefix for all texture filenames.
	*/
	STPWorldSplattingAgent(string prefix) : TDLParser(STPWorldSplattingAgent::readTDL()) {
		using namespace SuperTerrainPlus::STPDiversity;

		array<pair<unsigned int, const char*>, STPWorldSplattingAgent::TextureCount> IndexedFilename;
		//load up indices
		unsigned int index = 0u;
		generate(IndexedFilename.begin(), IndexedFilename.end(), [&index]() {
			const unsigned int i = index++;
			return make_pair(i, STPWorldSplattingAgent::Filename[i]);
		});

		//load all texture from the file system
		for_each(std::execution::par, IndexedFilename.cbegin(), IndexedFilename.cend(),
			[&texArr = this->LoadedData, &prefix](const auto& filename) {
			//currently we only worry about 3 channels, since all our images don't have alpha channel.
			//loading texture from file system can be done in parallel
			//insertion into the container needs to be safe.
			texArr[filename.first] = STPTextureStorage(prefix + '/' + filename.second, 3);
		});

		//create group
		STPTextureDatabase::STPTextureDescription tex_desc;
		tex_desc.Dimension = uvec2(1024u);
		tex_desc.PixelFormat = GL_UNSIGNED_BYTE;
		tex_desc.ChannelFormat = GL_RGB;
		tex_desc.InteralFormat = GL_RGB8;
		const STPTextureInformation::STPTextureGroupID x1024_rgb = this->Database.addGroup(tex_desc);

		//build texture splatting rules
		const STPTextureDefinitionLanguage::STPTextureVariable textureName = this->TDLParser(this->Database);
		//build database with texture data
		for (unsigned int i = 0u; i < STPWorldSplattingAgent::TextureCount; i++) {
			//grab the texture ID using the texture name
			const string_view currTexFile(STPWorldSplattingAgent::Filename[i]);
			//our filename always follows this pattern: (texture name)_(type).(suffix), we can search using that
			const STPTextureInformation::STPTextureID currTexID = textureName.at(currTexFile.substr(0ull, currTexFile.find_first_of('_')));
			
			this->Database.addMap(currTexID, STPTextureType::Albedo, x1024_rgb, this->LoadedData[i].texture());
		}
	}

	STPWorldSplattingAgent(const STPWorldSplattingAgent&) = delete;

	STPWorldSplattingAgent(STPWorldSplattingAgent&&) = delete;

	STPWorldSplattingAgent& operator=(const STPWorldSplattingAgent&) = delete;

	STPWorldSplattingAgent& operator=(STPWorldSplattingAgent&&) = delete;

	~STPWorldSplattingAgent() = default;

};

STPWorldManager::STPWorldManager(string tex_filename_prefix, STPEnvironment::STPConfiguration& settings) :
	SharedProgram(settings.getChunkSetting()), WorldSetting(std::move(settings)), 
	Texture(make_unique<STPWorldManager::STPWorldSplattingAgent>(tex_filename_prefix)), linkStatus(false) {
	if (!this->WorldSetting.validate()) {
		throw invalid_argument("World settings are not valid.");
	}
}

STPWorldManager::~STPWorldManager() = default;

void STPWorldManager::linkProgram(void* indirect_cmd) {
	this->linkStatus = false;
	//error checking
	if (!this->BiomeFactory) {
		throw invalid_argument("Biome factory not attached.");
	}

	const STPEnvironment::STPChunkSetting& chunk_settings = this->WorldSetting.getChunkSetting();
	//create generator and storage unit first
	this->ChunkGenerator.emplace(
		chunk_settings,
		this->WorldSetting.getHeightfieldSetting(),
		*this->DiversityGenerator,
		STPChunkProvider::calculateMaxConcurrency(chunk_settings.RenderedChunk, chunk_settings.FreeSlipChunk));
	this->ChunkStorage.emplace();
	//create provider using generator and storage unit
	this->ChunkProvider.emplace(chunk_settings, *this->ChunkStorage, *this->BiomeFactory, *this->ChunkGenerator);
	//create manager using provider
	this->ChunkManager.emplace(*this->ChunkProvider, *this->TextureFactory);
	//create renderer using manager
	this->WorldRenderer.emplace(this->WorldSetting.getMeshSetting(), *this->ChunkManager, indirect_cmd);

	this->linkStatus = true;
}

STPWorldManager::operator bool() const {
	return this->linkStatus;
}

SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPDatabaseView STPWorldManager::getTextureDatabase() const {
	return this->Texture->Database.visit();
}

const STPEnvironment::STPConfiguration& STPWorldManager::getWorldSetting() const {
	return this->WorldSetting;
}

const STPChunkManager& STPWorldManager::getChunkManager() const {
	return *this->ChunkManager;
}

const STPProcedural2DINF& STPWorldManager::getChunkRenderer() const {
	return *this->WorldRenderer;
}