//TEMPLATE DEFINITION FOR SCENE PIPELINE, DO NOT INCLUDE MANUALLY
#ifdef _STP_SCENE_PIPELINE_H_

#include <type_traits>

template<class Obj, typename ...Arg>
Obj* SuperTerrainPlus::STPRealism::STPScenePipeline::add(Arg&&... arg) {
	using std::is_base_of_v;
	using std::is_same_v;
	using std::unique_ptr;
	using std::make_unique;
	using std::forward;
	
	STPSceneGraph& scene_graph = this->SceneComponent;
	//Check the base class to determine the type of this rendering component
	if constexpr (is_base_of_v<STPSceneObject::STPOpaqueObject<false>, Obj>) {
		//it is an opaque object (don't care if it casts shadow or not.)
		//then insert into the opaque object list
		Obj* const opaque_obj_ptr = new Obj(forward<Arg>(arg)...);
		//manage the pointer by internal memory
		scene_graph.OpaqueObjectDatabase.emplace_back(opaque_obj_ptr);

		if constexpr (is_base_of_v<STPSceneObject::STPOpaqueObject<true>, Obj>) {
			//it is also a shadow casting opaque object, add to a subset list
			scene_graph.ShadowOpaqueObject.emplace_back(opaque_obj_ptr);

			//now configure this shadow-casting object with each depth configuration
			const STPShaderManager* const depth_shader = this->getDepthShader();
			for (const auto depth_config : scene_graph.UniqueLightSpaceSize) {
				opaque_obj_ptr->addDepthConfiguration(depth_config, depth_shader);
			}
		}

		return opaque_obj_ptr;
	}

	if constexpr (is_base_of_v<STPSceneObject::STPEnvironmentObject, Obj>) {
		Obj* const env_obj_ptr = new Obj(forward<Arg>(arg)...);
		scene_graph.EnvironmentObjectDatabase.emplace_back(env_obj_ptr);

		return env_obj_ptr;
	}

	if constexpr (is_base_of_v<STPSceneLight, Obj>) {
		unique_ptr<Obj> light_ptr_managed = make_unique<Obj>(forward<Arg>(arg)...);
		//Before record to the database, add to the scene renderer.
		//This function might thrown an exception in case addition to the renderer fails.
		this->addLight(*light_ptr_managed.get());

		//unique_ptr does not allow casting the underlying pointer, release it and re-create a new one.
		Obj* const light_ptr = light_ptr_managed.release();
		scene_graph.LightDatabase.emplace_back(light_ptr);
		//if this light casts shadow, we add to a subset list
		if (light_ptr->getLightShadow()) {
			//For now our system guarantees that if the light is initialised as a shadow-casting light it will stay.
			//We don't need to worry about whether light becomes a non shadow-casting light later. And vice-versa.
			scene_graph.ShadowLight.emplace_back(light_ptr);
		}

		return light_ptr;
	}

	if constexpr (is_same_v<STPAmbientOcclusion, Obj>) {
		return &scene_graph.AmbientOcclusionObject.emplace(forward<Arg>(arg)...);
	}

	if constexpr (is_same_v<STPPostProcess, Obj>) {
		return &scene_graph.PostProcessObject.emplace(forward<Arg>(arg)...);
	}
}

#endif//_STP_SCENE_PIPELINE_H_