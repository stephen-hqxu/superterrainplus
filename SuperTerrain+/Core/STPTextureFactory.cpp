#include <SuperTerrain+/World/Diversity/Texture/STPTextureFactory.h>

//Import implementation
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

#include <algorithm>

using namespace SuperTerrainPlus::STPDiversity;

using std::unordered_map;
using std::unique_ptr;
using std::make_unique;

void STPTextureFactory::formatRegion(const STPTextureDatabase::STPTypeMappingView& type_mapping, const STPTextureDatabase::STPGroupView& group_mapping) {
	//allocate temp host memory for region as cache
	const size_t regionSize = type_mapping.size() * static_cast<std::underlying_type_t<STPTextureType>>(STPTextureType::TYPE_COUNT);
	unique_ptr<STPTextureInformation::STPRegion[]> regionCache = make_unique<STPTextureInformation::STPRegion[]>(regionSize);
	//zero init the region, unsed region must be denoted by nullptr
	std::fill_n(regionCache.get(), regionSize, nullptr);

	
}

STPTextureFactory::STPTextureFactory(const STPTextureSplatBuilder& builder, const STPTextureDatabase& database) {
	
}