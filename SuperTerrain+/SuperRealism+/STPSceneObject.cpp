#include <SuperRealism+/Scene/STPSceneObject.h>

//Error
#include <SuperTerrain+/Exception/STPMemoryError.h>

//System
#include <algorithm>

using std::vector;
using std::binary_search;
using std::lower_bound;

using namespace SuperTerrainPlus::STPRealism;

inline auto STPSceneObject::STPDepthRendererGroup::getKeyLocation(unsigned int light_space_count) const {
	return lower_bound(this->LightSpaceSize.cbegin(), this->LightSpaceSize.cend(), light_space_count);
}

bool STPSceneObject::STPDepthRendererGroup::exist(unsigned int light_space_count) const {
	return binary_search(this->LightSpaceSize.cbegin(), this->LightSpaceSize.cend(), light_space_count);
}

STPPipelineManager& STPSceneObject::STPDepthRendererGroup::addGroup(unsigned int light_space_count) {
	const auto key_loc = this->getKeyLocation(light_space_count);
	if (*key_loc == light_space_count) {
		throw STPException::STPMemoryError("Another depth renderer group with the same configuration has been added previously");
	}

	//insert the new key while maintaining sorted order, get the index to locate the pipeline
	const size_t index = this->LightSpaceSize.insert(key_loc, light_space_count) - this->LightSpaceSize.begin();
	return *this->LightSpaceDepthRenderer.emplace(this->LightSpaceDepthRenderer.cbegin() + index);
}

STPPipelineManager& STPSceneObject::STPDepthRendererGroup::findGroup(unsigned int light_space_count) {
	const size_t index = this->getKeyLocation(light_space_count) - this->LightSpaceSize.cbegin();

	return this->LightSpaceDepthRenderer[index];
}