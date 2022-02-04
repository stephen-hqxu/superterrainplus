#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Component
#include "./Scene/STPSceneObject.h"
#include "./Scene/STPSceneLight.h"
#include "./Scene/Component/STPPostProcess.h"
//Lighting
#include "./Environment/STPLightSetting.h"
//Camera
#include "./Utility/Camera/STPCamera.h"
//GL Object
#include "./Object/STPTexture.h"
#include "./Object/STPBuffer.h"

//Container
#include <vector>
#include <memory>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPScenePipeline is a master rendering pipeline for the terrain engine.
	 * It manages all rendering components and issues rendering task automatically.
	 * It is recommended to have only one scene pipeline for one context.
	 * To allow rendering with the scene pipeline, a valid GL function header needs to be included before including this header.
	*/
	class STP_REALISM_API STPScenePipeline {
	public:

		/**
		 * @brief STPShadowMapFilter defines filtering technologies used for post-process shadow maps.
		*/
		enum class STPShadowMapFilter : unsigned char {
			//Nearest-Neighbour filter, shadow value is read from the nearest pixel.
			Nearest = 0x00u,
			//Bilinear filter, shadow value is read from its neighbours and linearly interpolated.
			Bilinear = 0x01u,
			//Percentage-Closer filter, it attempts to smooth the edge of the shadow using a blur kernel.
			PCF = 0x02u,
			//Multi-Sampled Variance Shadow Mapping, it uses variance to estimate the likelihood of a pixel that should have shadow 
			//after having the shadow map blurred, and also it is optionally multi-sampled.
			MSVSM = 0x03u,
			//Percentage-Closer Soft Shadow
			PCSS = 0x04u,
			//Exponential Shadow Mapping
			ESM = 0x05u
		};

		/**
		 * @brief STPSceneShaderCapacity controls declared array length when compiling scene shaders.
		 * Using large limits allow more flexible control to adding and removing rendering components to the scene later,
		 * using small limits save memory if user finds configure the scene dynamically unnecessary.
		 * All capacity settings are specified in terms of the number of element.
		*/
		struct STPSceneShaderCapacity {
		public:

			//The maximum number of environment light
			size_t EnvironmentLight;
			//The maximum number of directional light that can cast shadow.
			size_t DirectionalLightShadow;

			//The maximum number of light space matrix, as a 4 by 4 matrix of floats.
			size_t LightSpaceMatrix;
			//The maximum number of plane that divides light frustum into subfrusta, as float.
			size_t LightFrustumDivisionPlane;

		};

		/**
		 * @brief STPSceneShadowInitialiser specifies settings for scene shadow pipeline.
		*/
		struct STPSceneShadowInitialiser {
		public:

			//Max bias and min bias
			glm::vec2 ShadowMapBias;
			//Specify the algorithm used to filter the shadow map.
			STPShadowMapFilter ShadowFilter;
		};

		/**
		 * @brief STPLightPropertyType indicates the type of light property to be selected.
		 * The corresponded data type for the property type is also documented.
		 * When calling functions using a specific property, make sure the data type supplied is correct, otherwise it will give compile-time error.
		*/
		enum class STPLightPropertyType : unsigned char {
			//The multiplier to the ambient light
			//Float
			AmbientStrength = 0x00u,
			//The multiplier to the diffuse light
			//Float
			DiffuseStrength = 0x01u,
			//The multiplier to the specular light
			//Float
			SpecularStrength = 0x02u,
			//The sampling coordinate to the light spectrum
			//No data
			SpectrumCoordinate = 0x03u,
			//Light direction for directional light
			//No data
			Direction = 0x04u
		};

	private:

		/**
		 * @brief STPSceneGraph contains all rendering components for a scene pipeline to be rendered.
		*/
		struct STPSceneGraph {
		public:

			//Scene graph
			//Object nodes
			std::vector<std::unique_ptr<STPSceneObject::STPOpaqueObject<false>>> OpaqueObjectDatabase;
			//This is a subset-view of opaque object database, a collection of opaque objects that can cast shadow.
			std::vector<STPSceneObject::STPOpaqueObject<true>*> ShadowOpaqueObject;

			//Light nodes
			std::vector<size_t> UniqueLightSpaceSize;
			std::vector<std::unique_ptr<STPSceneLight::STPEnvironmentLight<false>>> EnvironmentObjectDatabase;
			std::vector<STPSceneLight::STPEnvironmentLight<true>*> ShadowEnvironmentObject;

			//Post process node
			std::unique_ptr<STPPostProcess> PostProcessObject;

		};

		/**
		 * @brief STPSharedTexture contains texture data that are shared with the children components in the scene pipeline.
		*/
		struct STPSharedTexture {
		public:

			STPTexture DepthStencil;

			/**
			 * @brief Default initialise shared texture memory.
			*/
			STPSharedTexture();

			STPSharedTexture(STPSharedTexture&&) noexcept = default;

			STPSharedTexture& operator=(STPSharedTexture&&) noexcept = default;

			~STPSharedTexture() = default;

		};

		STPSceneShaderCapacity SceneMemoryCurrent;
		const STPSceneShaderCapacity SceneMemoryLimit;

		//Shared buffer between different scene processors.
		STPSharedTexture SceneTexture;
		STPSceneGraph SceneComponent;

		/**
		 * @brief STPCameraInformationMemory stores memory for camera information.
		*/
		class STPCameraInformationMemory;
		std::unique_ptr<STPCameraInformationMemory> CameraMemory;
		/**
		 * @brief STPShadowPipeline is a shadow manager that handles all light source that can cast shadow and
		 * provide pipeline for rendering opaque objects onto a shadow map.
		*/
		class STPShadowPipeline;
		std::unique_ptr<STPShadowPipeline> GeometryShadowPass;
		/**
		 * @brief STPGeometryBufferResolution is the final step in a deferred rendering pipeline.
		 * It manages all lights in the scene and processes all captured G-buffer and perform lighting calculations.
		 * It is recommended to have only one of this instance per GL context and it should be used along with the master rendering pipeline.
		*/
		class STPGeometryBufferResolution;
		std::unique_ptr<STPGeometryBufferResolution> GeometryLightPass;
		/**
		 * @brief STPSceneRenderMemory captures the previously rendered image into an internal memory.
		*/
		class STPSceneRenderMemory;
		std::unique_ptr<STPSceneRenderMemory> RenderMemory;

		/**
		 * @brief Check if this light can be added to this scene without running out of memory.
		 * @param light_shadow The pointer to the shadow instance, or nullptr. Note that this light should not be added to the scene prior to this function call.
		 * If this light cannot be added, exception is thrown.
		 * This function always assumes a non-shadow casting light will be added.
		*/
		void canLightBeAdded(const STPSceneLight::STPEnvironmentLight<true>*) const;

		/**
		 * @brief For a newly added light, allocate light memory and flush light settings to the scene pipeline shader.
		 * This function does not thrown any error if the result of adding this light causes memory overflow, which results in UB.
		 * @param light The pointer to the newly added light.
		 * This light must have been added to the scene prior to this function call.
		 * @param light_shadow The pointer to the shadow instance of the light.
		 * The pointer can be null if this light does not cast shadow.
		*/
		void addLight(const STPSceneLight::STPEnvironmentLight<false>&, const STPSceneLight::STPEnvironmentLight<true>*);

	public:

		/**
		 * @brief STPScenePipelineLog contains logs from compilations of scene pipeline shaders.
		*/
		struct STPScenePipelineLog {
		public:

			/**
			 * @brief STPGeometryBufferResolutionLog stores log for the lighting pipeline. 
			*/
			struct STPGeometryBufferResolutionLog {
			public:

				STPScreen::STPScreenLog QuadShader;
				STPLogStorage<2ull> LightingShader;

			} GeometryBufferResolution;

		};

		/**
		 * @brief Initialise an empty scene pipeline.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		 * @param shader_cap The pointer to a struct that defines the maximum memory to be allocated for each array in the shader.
		 * @param shadow_init The pointer to the configurations for shadow rendering pipeline in the scene.
		 * @param log The pointer to log to output the initial compilation results for scene pipeline.
		*/
		STPScenePipeline(const STPCamera&, const STPSceneShaderCapacity&, const STPSceneShadowInitialiser&, STPScenePipelineLog&);

		STPScenePipeline(const STPScenePipeline&) = delete;

		STPScenePipeline(STPScenePipeline&&) = delete;

		STPScenePipeline& operator=(const STPScenePipeline&) = delete;

		STPScenePipeline& operator=(STPScenePipeline&&) = delete;

		~STPScenePipeline();

		/**
		 * @brief Get information about the amount of memory being used by the scene pipeline currently.
		 * @return The pointer to the scene memory usage.
		*/
		const STPSceneShaderCapacity& getMemoryUsage() const;

		/**
		 * @brief Get the information about the maximum amount of memory declared and allocated for the scene pipeline.
		 * @return The pointer to the scene max memory usage.
		*/
		const STPSceneShaderCapacity& getMemoryLimit() const;

		/**
		 * @brief Locate the index of a given light that is added to the scene graph.
		 * @param light The pointer to the light.
		 * @return The index of this light in the scene graph array.
		 * If the light is not registered with the scene, exception is thrown.
		 * This index is valid until the light is removed from the scene, adding new lights won't cause the index to be invalidated.
		*/
		size_t locateLight(const STPSceneLight::STPEnvironmentLight<false>*) const;

		/**
		 * @brief Add a rendering component to the scene pipeline.
		 * @tparam Obj The type of the object.
		 * @tparam ...Arg Arguments for constructing the object.
		 * @param arg... The argument lists.
		 * @return The pointer to the newly constructed rendering component.
		 * This pointer is managed by the current scene pipeline.
		 * If the object type is not supported, operation is ignored.
		*/
		template<class Obj, typename... Arg>
		Obj* add(Arg&&...);

		/**
		 * @brief Specify clear values for the color buffers.
		 * @param color Specify the red, green, blue, and alpha values used when the color buffers are cleared. The initial values are all 0.
		*/
		void setClearColor(glm::vec4);

		/**
		 * @brief Set the rendering resolution.
		 * This will cause reallocation of all rendering buffer, such as G-buffer used in deferred shading and post-processing buffer.
		 * It should be considered as a very expensive operation.
		 * @param resolution The new resolution to be set.
		*/
		void setResolution(glm::uvec2);

		/**
		 * @brief Set the light property
		 * If the operation is invalid, nothing will be done and the function will return silently.
		 * @tparam Prop The light property to be set.
		 * Note that the operation is invalid if the given property is not applicable for this type of light.
		 * @tparam T The type of the property data. The type must be in-lined with the data type specified by the type.
		 * @param index The light index that uniquely identifies a light in the scene graph.
		 * Operation is invalid if the index does not correspond to a valid light in the scene.
		 * @param data The data supplied whenever it is applicable for a specific property.
		*/
		template<STPLightPropertyType Prop>
		void setLight(size_t);
		//-----------------------------------
		template<STPLightPropertyType Prop>
		void setLight(size_t, float);

		/**
		 * @brief Traverse the scene graph and render every component in sequential order.
		 * This function does not modify the state of any rendering component.
		 * Any update need to be called by the caller prior to rendering.
		 * Any pending async operations will be sync automatically by this function before rendering.
		*/
		void traverse();

	};

}
#include "STPScenePipeline.inl"
#endif//_STP_SCENE_PIPELINE_H_