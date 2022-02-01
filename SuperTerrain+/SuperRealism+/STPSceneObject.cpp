#include <SuperRealism+/Scene/STPSceneObject.h>

//Error
#include <SuperTerrain+/Exception/STPMemoryError.h>

using std::array;
using std::make_pair;

using namespace SuperTerrainPlus::STPRealism;

template<size_t GS>
bool STPSceneObject::STPDepthRenderGroup<GS>::exist(unsigned int light_space_count) const {
	return this->LightSpaceDatabase.find(light_space_count) != this->LightSpaceDatabase.cend();
}

template<size_t GS>
typename STPSceneObject::STPDepthRenderGroup<GS>::STPGroupMember& STPSceneObject::STPDepthRenderGroup<GS>::addGroup(unsigned int light_space_count) {
	//the values are default constructable
	auto [it, inserted] = this->LightSpaceDatabase.try_emplace(light_space_count);
	if (!inserted) {
		throw STPException::STPMemoryError("Another depth renderer group with the same configuration has been added previously");
	}

	//ok to return a new member
	return it->second;
}

template<size_t GS>
STPPipelineManager& STPSceneObject::STPDepthRenderGroup<GS>::findPipeline(unsigned int light_space_count) {
	return const_cast<STPPipelineManager&>(const_cast<const STPDepthRenderGroup*>(this)->findPipeline(light_space_count));
}

template<size_t GS>
inline const STPPipelineManager& STPSceneObject::STPDepthRenderGroup<GS>::findPipeline(unsigned int light_space_count) const {
	//error checking is not needed, we have informed the user for the UB
	//because this function might get called multiple times every frame, just to make it cheaper.
	return this->LightSpaceDatabase.find(light_space_count)->second.first;
}

//Explicit Instantiation
#define DEPTH_RENDER_GROUP(COUNT) template class STP_REALISM_API STPSceneObject::STPDepthRenderGroup<COUNT>
DEPTH_RENDER_GROUP(1ull);
DEPTH_RENDER_GROUP(2ull);