#include "STPWorldManager.h"

#include <exception>

using std::invalid_argument;
using std::make_unique;

using namespace STPDemo;
using namespace SuperTerrainPlus;

STPWorldManager::STPWorldManager() : SharedProgram() {
	this->linkStatus = false;
}

void STPWorldManager::attachSetting(STPEnvironment::STPConfiguration& settings) {
	//move
	this->WorldSetting = make_unique<STPEnvironment::STPConfiguration>(std::move(settings));
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
	this->ChunkGenerator = make_unique<STPCompute::STPHeightfieldGenerator>(
		chunk_settings,
		this->WorldSetting->getHeightfieldSetting(),
		*this->DiversityGenerator,
		STPChunkProvider::calculateMaxConcurrency(chunk_settings.RenderedChunk, chunk_settings.FreeSlipChunk));
	this->ChunkStorage = make_unique<STPChunkStorage>();
	//create provider using generator and storage unit
	this->ChunkProvider = make_unique<STPChunkProvider>(chunk_settings, *this->ChunkStorage, *this->BiomeFactory, *this->ChunkGenerator);
	//create manager using provider
	this->ChunkManager = make_unique<STPChunkManager>(*this->ChunkProvider);
	//create renderer using manager
	this->WorldRenderer = make_unique<STPProcedural2DINF>(this->WorldSetting->getMeshSetting(), *this->ChunkManager, indirect_cmd);

	this->linkStatus = true;
}

STPWorldManager::operator bool() const {
	return this->linkStatus;
}

const STPEnvironment::STPConfiguration* STPWorldManager::getWorldSetting() const {
	return this->WorldSetting.get();
}

const STPCompute::STPHeightfieldGenerator* STPWorldManager::getChunkGenerator() const {
	return this->ChunkGenerator.get();
}

const STPDiversity::STPBiomeFactory* STPWorldManager::getBiomeFactory() const {
	return this->BiomeFactory.get();
}

const STPChunkStorage* STPWorldManager::getChunkStorage() const {
	return this->ChunkStorage.get();
}

const STPChunkProvider* STPWorldManager::getChunkProvider() const {
	return this->ChunkProvider.get();
}

const STPChunkManager* STPWorldManager::getChunkManager() const {
	return this->ChunkManager.get();
}

const STPProcedural2DINF* STPWorldManager::getChunkRenderer() const {
	return this->WorldRenderer.get();
}