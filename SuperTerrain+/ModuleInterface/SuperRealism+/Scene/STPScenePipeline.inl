//TEMPLATE DEFINITION FOR SCENE PIPELINE, DO NOT INCLUDE MANUALLY
#ifdef _STP_SCENE_PIPELINE_H_

#include <type_traits>

template<class Obj, typename ...Arg>
inline Obj& SuperTerrainPlus::STPRealism::STPScenePipeline::STPSceneInitialiser::add(Arg&&... arg) {
	using std::is_base_of_v;
	using std::is_same_v;
	using std::unique_ptr;
	using std::make_unique;
	using std::forward;
	using std::move;
	
	//Check the base class to determine the type of this rendering component
	if constexpr (is_base_of_v<STPSceneObject::STPOpaqueObject<false>, Obj>) {
		//it is an opaque object (don't care if it casts shadow or not.)
		//then insert into the opaque object list
		Obj* const opaque_obj_ptr = new Obj(forward<Arg>(arg)...);
		//manage the pointer by internal memory
		this->InitialiserComponent.OpaqueObjectDatabase.emplace_back(opaque_obj_ptr);

		if constexpr (is_base_of_v<STPSceneObject::STPOpaqueObject<true>, Obj>) {
			//it is also a shadow casting opaque object, add to a subset list
			this->InitialiserComponent.ShadowOpaqueObject.emplace_back(opaque_obj_ptr);
		}

		return *opaque_obj_ptr;
	}

	//using the same logic for the rests
	if constexpr (is_base_of_v<STPSceneLight::STPEnvironmentLight<false>, Obj>) {
		if (this->InitialiserComponent.EnvironmentObjectDatabase.size() == 1ull) {
			throw STPException::STPUnsupportedFunctionality("The system currently only supports one light that can cast shadow, "
				"this will be supported in the future release");
		}
		Obj* const env_obj_ptr = new Obj(forward<Arg>(arg)...);
		this->InitialiserComponent.EnvironmentObjectDatabase.emplace_back(env_obj_ptr);

		if constexpr (is_base_of_v<STPSceneLight::STPEnvironmentLight<true>, Obj>) {
			this->InitialiserComponent.ShadowEnvironmentObject.emplace_back(env_obj_ptr);

			this->LightSpaceCount += static_cast<unsigned int>(env_obj_ptr->getLightShadow().lightSpaceDimension());
		}

		return *env_obj_ptr;
	}

	if constexpr (is_same_v<STPPostProcess, Obj>) {
		this->InitialiserComponent.PostProcessObject = make_unique<Obj>(forward<Arg>(arg)...);

		return *this->InitialiserComponent.PostProcessObject;
	}
}

#endif//_STP_SCENE_PIPELINE_H_