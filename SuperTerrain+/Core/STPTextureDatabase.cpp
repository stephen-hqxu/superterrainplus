#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>
//Error
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

#include <algorithm>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using glm::uvec2;

using std::make_pair;

size_t STPTextureDatabase::IDAccumulator = 0ull;

template<typename ID, class S, class M>
STPTextureDatabase::STPTextureDataView<ID, S> STPTextureDatabase::sortView(const M& mapping) {
	STPTextureDataView<ID, S> mappingTable(mapping.size());
	//put the mapping into another data structure
	std::transform(mapping.cbegin(), mapping.cend(), mappingTable.begin(), [](const auto& elem) {
		return make_pair(elem.first, const_cast<const S*>(&(elem.second)));
		});

	//sort the view
	std::sort(mappingTable.begin(), mappingTable.end(), [](const auto& val1, const auto& val2) {
		return val1.first < val2.first;
		});

	return mappingTable;
}

const STPTextureDatabase::STPTypeInformation& STPTextureDatabase::getTypeMapping(STPTextureID id) const {
	auto it = this->TextureTypeMapping.find(id);
	if (it == this->TextureTypeMapping.cend()) {
		throw STPException::STPMemoryError("Texture ID is not found in the texture database");
	}
	return it->second;
}

STPTextureDatabase::STPTypeMappingView STPTextureDatabase::sortTypeMapping() const {
	return STPTextureDatabase::sortView<STPTextureID, STPTypeInformation>(this->TextureTypeMapping);
}

const STPTextureDatabase::STPTextureDescription& STPTextureDatabase::getGroupDescription(STPTextureGroupID id) const {
	auto it = this->TextureGroupRecord.find(id);
	if (it == this->TextureGroupRecord.cend()) {
		throw STPException::STPMemoryError("No group can be found in the database with the given group ID");
	}
	return it->second;
}

STPTextureDatabase::STPGroupView STPTextureDatabase::sortGroup() const {
	return STPTextureDatabase::sortView<STPTextureGroupID, STPTextureDescription>(this->TextureGroupRecord);
}

const void* STPTextureDatabase::operator()(STPTextureID id, STPTextureType type) const {
	//get all texture types available for this texture ID
	const STPTypeInformation& typeMapping = this->getTypeMapping(id);
	auto groupMappingit = typeMapping.find(type);
	if (groupMappingit == typeMapping.cend()) {
		throw STPException::STPMemoryError("The current texture ID has no associated texture type");
	}

	//we got the group ID, find the texture
	return groupMappingit->second.first;
}

STPTextureDatabase::STPTextureGroupID STPTextureDatabase::addGroup(const STPTextureDescription& desc) {
	//emplace a new group into the record
	//the current value of the accumulator will be the next value assigned to the new group
	auto [it, inserted] = this->TextureGroupRecord.try_emplace(static_cast<STPTextureGroupID>(STPTextureDatabase::IDAccumulator++), desc);
	//accumulator is unique therefore insertion will be guaranteed to take place
	//if the assertion fails it means something is wrong, probably the accumulator got hacked by user...
	assert(inserted);

	return it->first;
}

STPTextureDatabase::STPTextureID STPTextureDatabase::addTexture() {
	//add a new texture ID to the type mapping
	auto [it, inserted] = this->TextureTypeMapping.try_emplace(static_cast<STPTextureID>(STPTextureDatabase::IDAccumulator++));
	//insertion should be successful
	assert(inserted);

	return it->first;
}

bool STPTextureDatabase::addTextureData(STPTextureID texture_id, STPTextureType type, STPTextureGroupID group_id, const void* texture_data) {
	try {
		//make sure texture group is valid
		this->getGroupDescription(group_id);

		//make sure texture ID exists before performing insertion
		//insert only if type does not exist
		//get all texture data assoicated
		if (STPTypeInformation& typeData = const_cast<STPTypeInformation&>(this->getTypeMapping(texture_id));
			!typeData.try_emplace(type, texture_data, group_id).second) {
			return false;
		}
	}
	catch (...) {
		//safely ignore all exceptions
		return false;
	}

	return true;
}