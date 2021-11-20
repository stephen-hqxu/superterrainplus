#include "STPWorldManager.h"

//System
#include <exception>
#include <array>
#include <algorithm>
#include <execution>

//Texture Loader
#include "../Helpers/STPTextureStorage.h"

using namespace STPDemo;
using namespace SuperTerrainPlus;

using std::invalid_argument;
using std::make_optional;
using std::make_unique;

using std::array;
using std::string;
using std::pair;
using std::make_pair;
using std::for_each;
using std::generate;

using glm::ivec2;

class STPWorldManager::STPExternalTextureManager {
private:

	constexpr static size_t TextureCount = 6ull;
	//All loaded texture data
	STPTextureStorage LoadedData[STPExternalTextureManager::TextureCount];
	//List of all texture name
	constexpr static char* Filename[] = {
		"granite_color.png",
		"grass_color.png",
		"moss_color.png",
		"sand_color.png",
		"soil_color.png",
		"stone_color.png"
	};

public:

	/**
	 * @brief Init STPTextureStorage.
	 * @param prefix The filename prefix for all texture filenames.
	*/
	STPExternalTextureManager(string prefix) {
		array<pair<unsigned int, const char*>, STPExternalTextureManager::TextureCount> IndexedFilename;
		//load up indices
		unsigned int index = 0u;
		generate(IndexedFilename.begin(), IndexedFilename.end(), [&index]() {
			const unsigned int i = index++;
			return make_pair(i, STPExternalTextureManager::Filename[i]);
		});

		//load all texture from the file system
		for_each(std::execution::par, IndexedFilename.cbegin(), IndexedFilename.cend(),
			[&texArr = this->LoadedData, &prefix](const auto& filename) {
			//currently we only worry about 3 channels, since all our images don't have alpha channel.
			//loading texture from file system can be done in parallel
			//insertion into the container needs to be safe.
			texArr[filename.first] = STPTextureStorage(prefix + '/' + filename.second, 3);
		});
	}

	STPExternalTextureManager(const STPExternalTextureManager&) = delete;

	STPExternalTextureManager(STPExternalTextureManager&&) = delete;

	STPExternalTextureManager& operator=(const STPExternalTextureManager&) = delete;

	STPExternalTextureManager& operator=(STPExternalTextureManager&&) = delete;

	~STPExternalTextureManager() = default;

};

STPWorldManager::STPWorldManager(string tex_filename_prefix) : SharedProgram(), Texture(make_unique<STPWorldManager::STPExternalTextureManager>(tex_filename_prefix)) {
	this->linkStatus = false;
}

STPWorldManager::~STPWorldManager() = default;

void STPWorldManager::attachSetting(STPEnvironment::STPConfiguration& settings) {
	//move
	this->WorldSetting.emplace(std::move(settings));
}

void STPWorldManager::linkProgram(void* indirect_cmd) {
	this->linkStatus = false;
	//error checking
	if (!this->WorldSetting) {
		throw invalid_argument("World settings not attached.");
	}
	if (!this->BiomeFactory) {
		throw invalid_argument("Biome factory not attached.");
	}

	const STPEnvironment::STPChunkSetting& chunk_settings = this->WorldSetting->getChunkSetting();
	//create generator and storage unit first
	this->ChunkGenerator.emplace(
		chunk_settings,
		this->WorldSetting->getHeightfieldSetting(),
		*this->DiversityGenerator,
		STPChunkProvider::calculateMaxConcurrency(chunk_settings.RenderedChunk, chunk_settings.FreeSlipChunk));
	this->ChunkStorage.emplace();
	//create provider using generator and storage unit
	this->ChunkProvider.emplace(chunk_settings, *this->ChunkStorage, *this->BiomeFactory, *this->ChunkGenerator);
	//create manager using provider
	this->ChunkManager.emplace(*this->ChunkProvider);
	//create renderer using manager
	this->WorldRenderer.emplace(this->WorldSetting->getMeshSetting(), *this->ChunkManager, indirect_cmd);

	this->linkStatus = true;
}

STPWorldManager::operator bool() const {
	return this->linkStatus;
}

const STPEnvironment::STPConfiguration& STPWorldManager::getWorldSetting() const {
	return *this->WorldSetting;
}

const STPCompute::STPHeightfieldGenerator& STPWorldManager::getChunkGenerator() const {
	return *this->ChunkGenerator;
}

const STPDiversity::STPBiomeFactory& STPWorldManager::getBiomeFactory() const {
	return *this->BiomeFactory;
}

const STPChunkStorage& STPWorldManager::getChunkStorage() const {
	return *this->ChunkStorage;
}

const STPChunkProvider& STPWorldManager::getChunkProvider() const {
	return *this->ChunkProvider;
}

const STPChunkManager& STPWorldManager::getChunkManager() const {
	return *this->ChunkManager;
}

const STPProcedural2DINF& STPWorldManager::getChunkRenderer() const {
	return *this->WorldRenderer;
}