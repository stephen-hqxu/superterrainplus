#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Component
#include "../Renderer/STPPostProcess.h"
#include "STPSceneObject.h"
#include "STPSceneLight.h"
#include "../Renderer/STPPostProcess.h"
//Lighting
#include "../Environment/STPLightSetting.h"
//Camera
#include "../Utility/Camera/STPCamera.h"
//GL Object
#include "../Object/STPTexture.h"
#include "../Object/STPBuffer.h"

#include "../Utility/STPShadowInformation.hpp"

//Container
#include <list>
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
			std::vector<std::unique_ptr<STPSceneLight::STPEnvironmentLight<false>>> EnvironmentObjectDatabase;
			std::vector<STPSceneLight::STPEnvironmentLight<true>*> ShadowEnvironmentObject;

			//Post process node
			std::unique_ptr<STPPostProcess> PostProcessObject;

			STPSceneGraph() = default;

			STPSceneGraph(STPSceneGraph&&) noexcept = default;

			STPSceneGraph& operator=(STPSceneGraph&&) noexcept = default;

			~STPSceneGraph() = default;

		};

		/**
		 * @brief STPSceneShadowInitialiser specifies settings for scene shadow.
		*/
		struct STPSceneShadowInitialiser {
		protected:

			friend class STPScenePipeline;

			//Increment when a shadow-casting light is added.
			unsigned int LightSpaceCount = 0u;

		public:

			//Set the global resolution of the shadow map.
			//For performance consideration, the resolution of shadow maps in all shadow-casting light will be the same.
			glm::uvec2 ShadowMapResolution;
			//Max bias and min bias
			glm::vec2 ShadowMapBias;
			//Specify the algorithm used to filter the shadow map.
			STPShadowMapFilter ShadowFilter;
		};

		/**
		 * @brief STPSceneInitialiser pre-setup environment for scene pipeline.
		 * It helps building up a scene graph and passes to the scene pipeline.
		*/
		class STP_REALISM_API STPSceneInitialiser : public STPSceneShadowInitialiser {
		private:

			friend class STPScenePipeline;

			STPSceneGraph InitialiserComponent;

		public:

			STPSceneInitialiser() = default;

			~STPSceneInitialiser() = default;

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
			Obj& add(Arg&&...);

			/**
			 * @brief Generate a shadow initialiser based on the current setup.
			 * @return This variable can be used to initialise shadow-casting opaque objects after all shadow casting lights are added
			 * to the scene initialiser.
			 * The returned shadow information does not change unless more shadow-casting lights are added.
			*/
			STPShadowInformation shadowInitialiser() const;

		};

	private:

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
		 * @param init The pointer to the scene pipeline initialiser.
		 * After construction, the scene graph within the initialiser will be moved under the scene pipeline and become undefined.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		 * @param log The pointer to log to output the initial compilation results for scene pipeline.
		*/
		STPScenePipeline(STPSceneInitialiser&&, const STPCamera&, STPScenePipelineLog&);

		STPScenePipeline(const STPScenePipeline&) = delete;

		STPScenePipeline(STPScenePipeline&&) = delete;

		STPScenePipeline& operator=(const STPScenePipeline&) = delete;

		STPScenePipeline& operator=(STPScenePipeline&&) = delete;

		~STPScenePipeline();

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
		 * @brief Flush the light direction and spectrum coordinate.
		 * TODO: The system currently only supports a single light.
		 * @param direction The pointer to the new light direction.
		*/
		void updateLightStatus(const glm::vec3&);

		/**
		 * @brief Update the light property in the scene.
		 * @param ambient The pointer to the ambient light setting.
		 * @param directional The pointer to the directional light setting.
		 * @param shininess The specular power of the light.
		 * Shininess should be a material property rather than a light setting.
		 * TODO: Put this to a material system in the future.
		*/
		void setLightProperty(const STPEnvironment::STPLightSetting::STPAmbientLightSetting&, STPEnvironment::STPLightSetting::STPDirectionalLightSetting&, float);

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