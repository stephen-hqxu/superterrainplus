#include "STPWorldManager.h"

using std::make_unique;

using namespace SuperTerrainPlus;

STPWorldManager::STPWorldManager() {
	this->linkStatus = false;
}

void STPWorldManager::attachSettings(STPSettings::STPConfigurations* settings) {
	//make sure device memory is cleared otherwise it will result in undefined bebaviour
	settings->getHeightfieldSettings().omitDeviceAvailable();
	//copy
	this->WorldSettings = make_unique<STPSettings::STPConfigurations>(*settings);
	//and make managed copy device available
	this->WorldSettings->getHeightfieldSettings().makeDeviceAvailable();
}

void STPWorldManager::linkProgram(void* indirect_cmd) {
	try {
		const STPSettings::STPChunkSettings& chunk_settings = this->WorldSettings->getChunkSettings();
		//create generator and storage unit first
		this->ChunkGenerator = make_unique<STPCompute::STPHeightfieldGenerator>(
			this->WorldSettings->getSimplexNoiseSettings(),
			chunk_settings,
			this->WorldSettings->getHeightfieldSettings(),
			STPChunkProvider::calculateMaxConcurrency(chunk_settings.RenderedChunk, chunk_settings.FreeSlipChunk));
		this->ChunkStorage = make_unique<STPChunkStorage>();
		//create provider using generator and storage unit
		this->ChunkProvider = make_unique<STPChunkProvider>(chunk_settings, *this->ChunkStorage, *this->ChunkGenerator);
		//create manager using provider
		this->ChunkManager = make_unique<STPChunkManager>(*this->ChunkProvider);
		//create renderer using manager
		this->WorldRenderer = make_unique<STPProcedural2DINF>(this->WorldSettings->getMeshSettings(), *this->ChunkManager, indirect_cmd);
	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
		this->linkStatus = false;
	}
	this->linkStatus = true;
}

STPWorldManager::operator bool() const {
	return this->linkStatus;
}

const STPSettings::STPConfigurations* STPWorldManager::getWorldSettings() const {
	return this->WorldSettings.get();
}

const STPCompute::STPHeightfieldGenerator* STPWorldManager::getChunkGenerator() const {
	return this->ChunkGenerator.get();
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