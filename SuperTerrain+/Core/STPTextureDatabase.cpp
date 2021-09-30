#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Error
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using glm::uvec2;

using std::make_pair;

STPTextureDatabase::STPTextureGroup::STPTextureGroupID STPTextureDatabase::STPTextureGroup::ReferenceAccumulator = 0u;

STPTextureDatabase::STPTextureGroup::STPTextureGroup(const STPTextureDescription& desc)
	: GroupID(STPTextureGroup::ReferenceAccumulator), TextureProperty(desc) {
	//increment the accumulator
	STPTextureGroup::ReferenceAccumulator++;
}

size_t STPTextureDatabase::STPTextureGroup::STPKeyHasher::operator()(const STPTextureKey& key) const {
	using std::string;
	using std::hash;

	//because the texture type is a fixed size enum, it's easier to encode the string and hash the entire string just for once
	auto [id, type] = key;
	const string raw = id + "@@" + std::to_string(static_cast<std::underlying_type_t<STPTextureType>>(type)) + "@@";

	return hash<string>()(raw);
}

bool STPTextureDatabase::STPTextureGroup::add(STPTextureKey key, const void* texture) {
	return this->TextureDataRecord.try_emplace(key, texture).second;
}

const void* STPTextureDatabase::STPTextureGroup::operator[](STPTextureKey key) const {
	auto it = this->TextureDataRecord.find(key);
	if (it == this->TextureDataRecord.cend()) {
		throw STPException::STPMemoryError("Texture ID is not found in the requesting texture group");
	}
	return it->second;
}

const STPTextureDatabase::STPTypeGroupMapping& STPTextureDatabase::getTypeMapping(STPTextureID id) const {
	auto it = this->TextureTypeMapping.find(id);
	if (it == this->TextureTypeMapping.cend()) {
		throw STPException::STPMemoryError("Texture ID is not found in the texture database");
	}
	return it->second;
}

const STPTextureDatabase::STPTextureGroup& STPTextureDatabase::getGroup(STPTextureGroup::STPTextureGroupID id) const {
	auto it = this->TextureGroupRecord.find(id);
	if (it == this->TextureGroupRecord.cend()) {
		throw STPException::STPMemoryError("No group can be found in the database with the given group ID");
	}
	return it->second;
}

const void* STPTextureDatabase::operator()(STPTextureID id, STPTextureType type) const {
	//get all texture types available for this texture ID
	const STPTypeGroupMapping& typeMapping = this->getTypeMapping(id);
	auto groupMappingit = typeMapping.find(type);
	if (groupMappingit == typeMapping.cend()) {
		throw STPException::STPMemoryError("The current texture ID has no associated texture type");
	}

	//we got the group ID, find the group in the record
	const STPTextureGroup::STPTextureGroupID groupID = groupMappingit->second;
	const STPTextureGroup& group = this->getGroup(groupID);

	//get the texture data
	return group[make_pair(id, type)];
}

STPTextureDatabase::STPTextureGroup::STPTextureGroupID STPTextureDatabase::addGroup(const STPTextureDescription& desc) {
	//emplace a new group into the record
	//the current value of the accumulator will be the next value assigned to the new group
	auto [it, inserted] = this->TextureGroupRecord.try_emplace(STPTextureGroup::ReferenceAccumulator, desc);
	//accumulator is unique therefore insertion will be guaranteed to take place
	//if the assertion fails it means something is wrong, probably the accumulator got hacked by user...
	assert(inserted);

	return it->second.GroupID;
}

bool STPTextureDatabase::addTexture(STPTextureID texture_id, STPTextureType type, STPTextureGroup::STPTextureGroupID group_id, const void* texture_data) {
	//if texture ID exists, get it; if not, insert a new mapping
	STPTypeGroupMapping& typeMapping = this->TextureTypeMapping[texture_id];
	
	//insert only if type does not exist
	if (!typeMapping.try_emplace(type, group_id).second) {
		return false;
	}
	
	//find the group
	STPTextureGroup& group = const_cast<STPTextureGroup&>(this->getGroup(group_id));
	return group.add(make_pair(texture_id, type), texture_data);
}