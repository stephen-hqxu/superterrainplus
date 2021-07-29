#pragma once
#include <Environment/STPConfiguration.h>

using namespace SuperTerrainPlus::STPEnvironment;

//STPConfiguration.h

bool STPConfiguration::validate() const {
	return this->ChunkSetting.validate()
		&& this->HeightfieldSetting.validate()
		&& this->MeshSetting.validate();
}

STPChunkSetting& STPConfiguration::getChunkSetting() {
	return this->ChunkSetting;
}

STPHeightfieldSetting& STPConfiguration::getHeightfieldSetting() {
	return this->HeightfieldSetting;
}

STPMeshSetting& STPConfiguration::getMeshSetting() {
	return this->MeshSetting;
}

//STPChunkSetting.h

STPChunkSetting::STPChunkSetting() : STPSetting() {
	//fill with defaults
	this->ChunkSize = glm::uvec2(0u);
	this->MapSize = glm::uvec2(0u);
	this->RenderedChunk = glm::uvec2(0u);
	this->ChunkOffset = glm::vec3(0.0f);
	this->ChunkScaling = 1.0f;
	this->MapOffset = glm::vec3(0.0f);
	this->FreeSlipChunk = glm::uvec2(0u);
}

bool STPChunkSetting::validate() const {
	return this->ChunkScaling > 0.0f
		&& this->FreeSlipChunk.x >= 1u
		&& this->FreeSlipChunk.y >= 1u;
}

//STPHeightfieldSetting.h

STPHeightfieldSetting::STPHeightfieldSetting() : STPRainDropSetting() {
	this->Seed = 0ull;
	this->Strength = 1.0f;
}

bool STPHeightfieldSetting::validate() const {
	static auto checkRange = [](float value, float lower, float upper) -> bool {
		return value >= lower && value <= upper;
	};
	//check the raindrop parameter plus also heightmap parameter
	return this->STPRainDropSetting::validate()
		&& this->Strength > 0.0f;
}

//STPMeshSetting.h

STPMeshSetting::STPTessellationSetting::STPTessellationSetting() {
	this->MaxTessLevel = 0.0f;
	this->MinTessLevel = 0.0f;
	this->FurthestTessDistance = 0.0f;
	this->NearestTessDistance = 0.0f;
}

bool STPMeshSetting::STPTessellationSetting::validate() const {
	return this->MaxTessLevel >= 0.0f
		&& this->MinTessLevel >= 0.0f
		&& this->FurthestTessDistance >= 0.0f
		&& this->NearestTessDistance >= 0.0f
		//range check
		&& this->MaxTessLevel >= this->MinTessLevel
		&& this->FurthestTessDistance >= this->NearestTessDistance;
}

STPMeshSetting::STPMeshSetting() {
	this->Altitude = 1.0f;
	this->LoDShiftFactor = 2.0f;
}

bool STPMeshSetting::validate() const {
	return this->Altitude > 0.0f
		&& this->LoDShiftFactor > 0.0f
		&& this->TessSetting.validate();
}