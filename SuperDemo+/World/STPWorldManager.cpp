#include "STPWorldManager.h"

//Error
#include <SuperTerrain+/Exception/STPInvalidArgument.h>

//System
#include <exception>
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
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDefinitionLanguage.h>

//GLAD
#include <glad/glad.h>

using namespace STPDemo;
using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

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

	constexpr static size_t TextureCount = 11ull;
	//All loaded texture data
	STPTextureStorage LoadedData[STPWorldSplattingAgent::TextureCount];
	//List of all texture name
	constexpr static char* Filename[] = {
		"darkrock_color.jpg",
		"darkrock_normal.jpg",
		"grass_color.jpg",
		"grass_normal.jpg",
		"mossrock_color.jpg",
		"mossrock_normal.jpg",
		"redrock_color.jpg",
		"redrock_normal.jpg",
		"sand_color.jpg",
		"sand_normal.jpg",
		"soil_color.jpg"
	};
	constexpr static char TDLFilename[] = "./Script/STPBiomeSplatRule.tdl";

	//Group ID recording
	STPTextureInformation::STPTextureGroupID x1024_srgb, x1024_rgb;

	/**
	 * @brief Determine the texture type based on the filename.
	 * @param filename The filename of the texture.
	 * @return The type of texture.
	*/
	static STPTextureType getType(const string_view& filename) {
		//find the string representation of the type
		const size_t type_start = filename.find('_') + 1ull,
			type_end = filename.find('.');
		const string_view typeStr = filename.substr(type_start, type_end - type_start);

		//convert string to texture type
		if (typeStr == "color") {
			return STPTextureType::Albedo;
		}
		if (typeStr == "normal") {
			return STPTextureType::Normal;
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
	STPWorldSplattingAgent(string prefix) {
		array<pair<unsigned int, const char*>, STPWorldSplattingAgent::TextureCount> IndexedFilename;
		//load up indices
		unsigned int index = 0u;
		generate(IndexedFilename.begin(), IndexedFilename.end(), [&index]() {
			const unsigned int i = index++;
			return make_pair(i, STPWorldSplattingAgent::Filename[i]);
		});

		//load all texture from the file system
		for_each(std::execution::par, IndexedFilename.cbegin(), IndexedFilename.cend(),
			[&texArr = this->LoadedData, &prefix](const auto& data) {
			const auto [index, filename] = data;
			const int channel = 3;

			//all our images don't have alpha channel.
			//loading texture from file system can be done in parallel
			//insertion into the container needs to be safe.
			texArr[index] = STPTextureStorage(prefix + '/' + filename, channel);
		});

		//create group
		STPTextureDatabase::STPTextureDescription tex_desc;
		tex_desc.Dimension = uvec2(1024u);
		tex_desc.PixelFormat = GL_UNSIGNED_BYTE;
		tex_desc.ChannelFormat = GL_RGB;
		tex_desc.InteralFormat = GL_SRGB8;
		this->x1024_srgb = this->Database.addGroup(tex_desc);
		tex_desc.InteralFormat = GL_RGB8;
		this->x1024_rgb = this->Database.addGroup(tex_desc);

		STPDiversity::STPTextureDefinitionLanguage TDLParser(*STPFile(STPWorldSplattingAgent::TDLFilename));
		//build texture splatting rules
		const STPTextureDefinitionLanguage::STPTextureVariable textureName = TDLParser(this->Database);
		//build database with texture data
		for (unsigned int i = 0u; i < STPWorldSplattingAgent::TextureCount; i++) {
			//grab the texture ID using the texture name
			const string_view currTexFile(STPWorldSplattingAgent::Filename[i]);
			//our filename always follows this pattern: (texture name)_(type).(suffix), we can search using that
			const STPTextureInformation::STPTextureID currTexID = textureName.at(currTexFile.substr(0ull, currTexFile.find_first_of('_')));

			STPTextureType texType = STPWorldSplattingAgent::getType(currTexFile);
			STPTextureInformation::STPTextureGroupID texGroup = this->x1024_rgb;
			//change group ID conditionally
			if (texType == STPTextureType::Albedo) {
				//color texture is usually in gamma space, we need to transform it to linear space.
				texGroup = this->x1024_srgb;
			}
			this->Database.addMap(currTexID, texType, texGroup, this->LoadedData[i].texture());
		}
	}

	STPWorldSplattingAgent(const STPWorldSplattingAgent&) = delete;

	STPWorldSplattingAgent(STPWorldSplattingAgent&&) = delete;

	STPWorldSplattingAgent& operator=(const STPWorldSplattingAgent&) = delete;

	STPWorldSplattingAgent& operator=(STPWorldSplattingAgent&&) = delete;

	~STPWorldSplattingAgent() = default;

	/**
	 * @brief Set the texture parameter for a all texture groups.
	 * @param factory The pointer to the texture factory where texture parameters will be set.
	 * @param anisotropy_filter The level of anisotropy filtering to be used for each texture.
	*/
	void setTextureParameter(const STPTextureFactory& factory, float anisotropy_filter) const {
		//get the TBO based on group ID, currently we only have one group
		const array<GLuint, 2ull> all_tbo = {
			factory[this->x1024_srgb], factory[this->x1024_rgb]
		};

		for (const auto tbo : all_tbo) {
			glTextureParameteri(tbo, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTextureParameteri(tbo, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTextureParameteri(tbo, GL_TEXTURE_WRAP_R, GL_REPEAT);
			glTextureParameteri(tbo, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTextureParameteri(tbo, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTextureParameterf(tbo, GL_TEXTURE_MAX_ANISOTROPY, anisotropy_filter);

			glGenerateTextureMipmap(tbo);
		}
	}

};

STPWorldManager::STPWorldManager(string tex_filename_prefix, STPEnvironment::STPConfiguration& settings, 
	const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) :
	SharedProgram(settings.ChunkSetting, simplex_setting), WorldSetting(std::move(settings)),
	Texture(make_unique<STPWorldManager::STPWorldSplattingAgent>(tex_filename_prefix)), linkStatus(false) {
	if (!this->WorldSetting.validate()) {
		throw invalid_argument("World settings are not valid.");
	}
}

STPWorldManager::~STPWorldManager() = default;

void STPWorldManager::linkProgram(float anisotropy) {
	this->linkStatus = false;
	//error checking
	if (!this->BiomeFactory) {
		throw invalid_argument("Biome factory not attached.");
	}
	if (!this->TextureFactory) {
		throw invalid_argument("Texture factory not attached.");
	}

	//finish up texture settings
	this->Texture->setTextureParameter(*this->TextureFactory, anisotropy);

	const STPEnvironment::STPChunkSetting& chunk_settings = this->WorldSetting.ChunkSetting;
	//create generator and storage unit first
	this->ChunkGenerator.emplace(
		chunk_settings,
		this->WorldSetting.HeightfieldSetting,
		*this->DiversityGenerator,
		//TODO: fix the occupancy calculator
		1u);
	//create the world pipeline
	STPWorldPipeline::STPPipelineSetup pipeStage;
	pipeStage.BiomemapGenerator = this->BiomeFactory.get();
	pipeStage.HeightfieldGenerator = &(*this->ChunkGenerator);
	pipeStage.SplatmapGenerator = this->TextureFactory.get();
	pipeStage.ChunkSetting = &chunk_settings;
	this->Pipeline.emplace(pipeStage);
	
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

STPWorldPipeline& STPWorldManager::getPipeline() {
	return *this->Pipeline;
}