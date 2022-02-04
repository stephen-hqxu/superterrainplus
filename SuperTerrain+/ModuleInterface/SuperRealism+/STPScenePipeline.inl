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
			for (const auto depth_config : scene_graph.UniqueLightSpaceSize) {
				opaque_obj_ptr->addDepthConfiguration(depth_config);
			}
		}

		return opaque_obj_ptr;
	}

	//using the same logic for light
	if constexpr (is_base_of_v<STPSceneLight::STPEnvironmentLight<false>, Obj>) {
		if (scene_graph.EnvironmentObjectDatabase.size() == 1ull) {
			throw STPException::STPUnsupportedFunctionality("The system currently only supports one light, "
				"this will be supported in the future release");
		}
		//can this light cast shadow?
		constexpr static bool isShadowLight = is_base_of_v<STPSceneLight::STPEnvironmentLight<true>, Obj>;

		unique_ptr<Obj> env_obj_managed = make_unique<Obj>(forward<Arg>(arg)...);
		//this validity checker may thrown exception, so we need to manage this memory
		if constexpr (isShadowLight) {
			this->canLightBeAdded(env_obj_managed.get());
		}
		else {
			this->canLightBeAdded(nullptr);
		}

		//unique_ptr does not allow casting the underlying pointer, release it and re-create a new one.
		Obj* const env_obj_ptr = env_obj_managed.release();
		scene_graph.EnvironmentObjectDatabase.emplace_back(env_obj_ptr);

		if constexpr (isShadowLight) {
			scene_graph.ShadowEnvironmentObject.emplace_back(env_obj_ptr);

			//for a newly added light, the scene pipeline need to do something else
			this->addLight(*env_obj_ptr, env_obj_ptr);
		}
		else {
			//not a shadow casting light, shadow instance can be null
			this->addLight(*env_obj_ptr, nullptr);
		}

		return env_obj_ptr;
	}

	if constexpr (is_same_v<STPPostProcess, Obj>) {
		scene_graph.PostProcessObject = make_unique<Obj>(forward<Arg>(arg)...);

		return scene_graph.PostProcessObject.get();
	}
}

#endif//_STP_SCENE_PIPELINE_H_