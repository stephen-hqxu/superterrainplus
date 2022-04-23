#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Component
#include "./Scene/STPSceneObject.h"
#include "./Scene/STPSceneLight.h"
#include "./Scene/Component/STPPostProcess.h"
#include "./Scene/Component/STPAmbientOcclusion.h"
#include "./Scene/Light/STPShadowMapFilter.hpp"
//Camera
#include "./Utility/Camera/STPCamera.h"

//Container
#include <vector>
#include <unordered_set>
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

		//An integer to identify a light added to the scene in the shader.
		typedef size_t STPLightIdentifier;

		/**
		 * @brief STPShadowMapFilterFunction is an adaptive shadow map filter manager for any shadow map filter.
		*/
		class STP_REALISM_API STPShadowMapFilterFunction {
		private:

			friend class STPScenePipeline;

			/**
			 * @brief Flush the shadow map filter settings to a given program.
			 * @param program The pointer to the program to be flushed.
			*/
			virtual void operator()(STPProgramManager&) const = 0;

		public:

			const STPShadowMapFilter Filter;

			//For most shadow map filters, this controls the max and min bias respectively.
			//The pixel is moved in the direction of the light.
			//The depth bias is applied to normalised depth, which is in the range [0,1].
			glm::vec2 DepthBias;
			//The pixels are moved in the direction of the surface normal.
			//The normal bias is applied to world space normal.
			glm::vec2 NormalBias;
			//Specifies the how the bias should scale with the far plane of light frustum for directional light shadow.
			//Higher value gives less bias for further far plane.
			float BiasFarMultiplier;

			//Amount to overlap when blending between cascades.
			//Set to a positive value to enable cascade blending, which makes the program faster but leaving sharp edges at cascade transition.
			//This option only applies to directional light shadow.
			float CascadeBlendArea;

			/**
			 * @brief Init a STPShadowMapFilterFunction.
			 * @param filter The type of shadow map filter.
			*/
			STPShadowMapFilterFunction(STPShadowMapFilter);

			STPShadowMapFilterFunction(const STPShadowMapFilterFunction&) = default;

			STPShadowMapFilterFunction(STPShadowMapFilterFunction&&) noexcept = default;

			STPShadowMapFilterFunction& operator=(const STPShadowMapFilterFunction&) = delete;

			STPShadowMapFilterFunction& operator=(STPShadowMapFilterFunction&&) = delete;

			virtual ~STPShadowMapFilterFunction() = default;

			/**
			 * @brief Check if all values for the filter are valid.
			 * @return True if all of them are valid.
			*/
			bool valid() const;

		};

		/**
		 * @brief STPShadowMapFilterKernel defines the kernel of a shadow map filter.
		*/
		template<STPShadowMapFilter Fil>
		struct STPShadowMapFilterKernel : public STPShadowMapFilterFunction {
		private:

			void operator()(STPProgramManager&) const override {};

		public:

			STPShadowMapFilterKernel();

			~STPShadowMapFilterKernel() = default;

		};

		/**
		 * @brief STPSceneShaderCapacity controls declared array length when compiling scene shaders.
		 * Using large limits allow more flexible control to adding and removing rendering components to the scene later,
		 * using small limits save memory if user finds configure the scene dynamically unnecessary.
		 * All capacity settings are specified in terms of the number of element.
		*/
		struct STPSceneShaderCapacity {
		public:

			size_t AmbientLight, DirectionalLight;

		};

	private:

		/**
		 * @brief STPSceneGraph contains all rendering components for a scene pipeline to be rendered.
		*/
		struct STPSceneGraph {
		public:

			//Scene graph
			//Object nodes
			std::vector<STPSceneObject::STPOpaqueObject<false>*> OpaqueObjectDatabase;
			//This is a subset-view of opaque object database, a collection of opaque objects that can cast shadow.
			std::vector<STPSceneObject::STPOpaqueObject<true>*> ShadowOpaqueObject;
			//Holding all objects that allow light to pass through.
			//Rendering these objects is a bit more complicated in a deferred renderer.
			std::vector<STPSceneObject::STPTransparentObject*> TransparentObjectDatabase;
			//A special object that contributes to the environment and does not have a solid body.
			std::vector<STPSceneObject::STPEnvironmentObject*> EnvironmentObjectDatabase;

			//Light nodes
			std::unordered_set<size_t> UniqueLightSpaceSize;
			std::vector<STPSceneLight*> ShadowLight;

			//Special effect nodes
			STPAmbientOcclusion* AmbientOcclusionObject = nullptr;
			STPPostProcess* PostProcessObject = nullptr;

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

		//The albedo colour to be cleared to for all off-screen framebuffer.
		glm::vec4 DefaultClearColor;

		/**
		 * @brief Get the shader used for performing additional operations during depth rendering.
		 * @return The pointer to the depth shader. Nullprt is returned if depth shader is unused.
		*/
		const STPShaderManager* getDepthShader() const;
		
		/**
		 * @brief Add a light to the scene pipeline.
		 * Exception is thrown if the scene pipeline has no more memory to hold more lights.
		 * @param light The pointer to the light that should be added.
		*/
		void addLight(STPSceneLight&);

	public:

		/**
		 * @brief STPScenePipelineInitialiser contains initialisers and logs from compilations of scene pipeline shaders.
		*/
		struct STPScenePipelineInitialiser {
		public:

			//Defines the maximum memory to be allocated for each array in the shader.
			STPSceneShaderCapacity ShaderCapacity;
			//Specifies shadow map filter function to be used in the scene.
			const STPShadowMapFilterFunction* ShadowFilter;

			//Pointer to lighting shader initialiser
			const STPScreen::STPScreenInitialiser* GeometryBufferInitialiser;

		};

		/**
		 * @brief Initialise an empty scene pipeline.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		 * @param scene_init The pointer to scene initialiser.
		*/
		STPScenePipeline(const STPCamera&, STPScenePipelineInitialiser&);

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
		 * @brief Add a rendering component to the scene pipeline.
		 * @tparam Obj The type of the object.
		 * @param object The pointer to the object to be added.
		 * The pointer is retained by the scene pipeline; unless the pipeline is destroyed or object is removed from it,
		 * the lifetime of this object should remain.
		*/
		template<class Obj>
		void add(Obj&);

		/**
		 * @brief Specify clear values for the colour buffers.
		 * @param colour Specify the red, green, blue, and alpha values used when the colour buffers are cleared. The initial values are all 0.
		*/
		void setClearColor(glm::vec4);

		/**
		 * @brief Specify should the rendering pipeline uses representative fragment testing.
		 * This feature requires extension GL_NV_representative_fragment_test.
		 * As of 09/03/2022, survey shows this extension is only available on NVIDIA Turing and Ampere architecture.
		 * @param val Enable flag. Set to true to enable this feature, false otherwise.
		 * @return A value reflecting if the feature has been turned on.
		 * This should usually be equal to `val`, and only be false all the time if the targeting system does not support this feature.
		*/
		bool setRepresentativeFragmentTest(bool);

		/**
		 * @brief Set the rendering resolution.
		 * This will cause reallocation of all rendering buffer, such as G-buffer used in deferred shading and post-processing buffer.
		 * It should be considered as a very expensive operation.
		 * @param resolution The new resolution to be set.
		*/
		void setResolution(glm::uvec2);

		/**
		 * @brief Set the region where the distant object starts to fade off.
		 * @param factor A multiplier to the camera far distance where the extinction zone starts.
		*/
		void setExtinctionArea(float) const;

		/**
		 * @brief Traverse the scene graph and render every component in sequential order.
		 * This function does not modify the state of any rendering component.
		 * Any update need to be called by the caller prior to rendering.
		 * Any pending async operations will be sync automatically by this function before rendering.
		*/
		void traverse();

	};

#define SHADOW_MAP_FILTER_DEF(FILT) \
template<> struct STP_REALISM_API STPScenePipeline::STPShadowMapFilterKernel<STPShadowMapFilter::FILT> : public STPScenePipeline::STPShadowMapFilterFunction

	SHADOW_MAP_FILTER_DEF(PCF) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPShadowMapFilterKernel();

		~STPShadowMapFilterKernel() = default;

		//Specifies the radius of the filter kernel.
		//Larger radius gives smoother shadow but also slower.
		unsigned int KernelRadius;
		//Specifies the distance between each sampling points.
		float KernelDistance;

	};

	SHADOW_MAP_FILTER_DEF(VSM) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPShadowMapFilterKernel();

		~STPShadowMapFilterKernel() = default;

		//The minimum variance value. 
		//This helps to reduce shadow acne effects when light direction is parallel to the surface.
		float minVariance;

		//Specifies the number of mipmap level to use.
		unsigned int mipmapLevel;
		//Specifies the anisotropy filter level. 
		float AnisotropyFilter;

	};

#undef SHADOW_MAP_FILTER_DEF

}
#include "STPScenePipeline.inl"
#endif//_STP_SCENE_PIPELINE_H_