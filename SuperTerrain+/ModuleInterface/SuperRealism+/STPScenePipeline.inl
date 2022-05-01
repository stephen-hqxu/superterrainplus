//TEMPLATE DEFINITION FOR SCENE PIPELINE, DO NOT INCLUDE MANUALLY
#ifdef _STP_SCENE_PIPELINE_H_

#include <type_traits>

#include <SuperTerrain+/Exception/STPMemoryError.h>

template<class Obj>
void SuperTerrainPlus::STPRealism::STPScenePipeline::add(Obj& object) {
	using std::is_base_of_v;
	using std::is_same_v;
	
	STPSceneGraph& scene_graph = this->SceneComponent;
	//Check the base class to determine the type of this rendering component
	if constexpr (is_base_of_v<STPSceneObject::STPOpaqueObject<false>, Obj>) {
		//it is an opaque object (don't care if it casts shadow or not.)
		//then insert into the opaque object list
		//manage the pointer by internal memory
		scene_graph.OpaqueObjectDatabase.emplace_back(&object);

		if constexpr (is_base_of_v<STPSceneObject::STPOpaqueObject<true>, Obj>) {
			//it is also a shadow casting opaque object, add to a subset list
			scene_graph.ShadowOpaqueObject.emplace_back(&object);

			//now configure this shadow-casting object with each depth configuration
			const STPShaderManager* const depth_shader = this->getDepthShader();
			for (const auto depth_config : scene_graph.UniqueLightSpaceSize) {
				object.addDepthConfiguration(depth_config, depth_shader);
			}
		}
		return;
	}

	if constexpr (is_base_of_v<STPSceneObject::STPTransparentObject, Obj>) {
		scene_graph.TransparentObjectDatabase.emplace_back(&object);
		return;
	}

	if constexpr (is_base_of_v<STPSceneObject::STPEnvironmentObject, Obj>) {
		scene_graph.EnvironmentObjectDatabase.emplace_back(&object);
		return;
	}

	if constexpr (is_base_of_v<STPSceneLight, Obj>) {
		//Before record to the database, add to the scene renderer.
		//This function might thrown an exception in case addition to the renderer fails.
		this->addLight(object);

		//if this light casts shadow, we add to a subset list
		if (object.getLightShadow()) {
			//For now our system guarantees that if the light is initialised as a shadow-casting light it will stay.
			//We don't need to worry about whether light becomes a non shadow-casting light later. And vice-versa.
			scene_graph.ShadowLight.emplace_back(&object);
		}
		return;
	}

	if constexpr (is_same_v<STPAmbientOcclusion, Obj>) {
		scene_graph.AmbientOcclusionObject = &object;
		return;
	}

	if constexpr (is_same_v<STPBidirectionalScattering, Obj>) {
		if (!this->hasMaterialLibrary) {
			throw STPException::STPMemoryError("Bidirectional scattering effect requires material data, "
				"however material library is not available in this scene pipeline instance");
		}
		scene_graph.BSDFObject = &object;
		return;
	}

	if constexpr (is_same_v<STPPostProcess, Obj>) {
		scene_graph.PostProcessObject = &object;
		return;
	}
}

#endif//_STP_SCENE_PIPELINE_H_