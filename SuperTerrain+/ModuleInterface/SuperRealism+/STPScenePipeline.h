#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Component
#include "./Scene/STPSceneObject.h"
#include "./Scene/STPSceneLight.h"
#include "./Scene/Component/STPPostProcess.h"
#include "./Scene/Component/STPAmbientOcclusion.h"
#include "./Scene/Component/STPBidirectionalScattering.h"
#include "./Scene/Light/STPShadowMapFilter.hpp"
#include "./Scene/STPMaterialLibrary.h"
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

		/* ------------------------------- Shading ------------------------------------ */

		/**
		 * @brief STPShadingModel specifies the shading equation used for scene lighting.
		*/
		enum class STPShadingModel : unsigned char {
			//One of the oldest but still widely used shading model.
			//Although it is not physically accurate, but the approximation of lighting gives reasonable image quality
			//First proposed by Bui Tuong Phong in 1975, and improved by James F. Blinn in 1977.
			BlinnPhong = 0x00u,
			//One of the most widely used shading model for physically-based rendering.
			//With the introduction of micro-facet model, it can simulate roughness and conservation of energy better.
			//Proposed by R. Cook and K. Torrance in 1981.
			CookTorrance = 0x01u
		};

		/**
		 * @brief STPShadingModelDescription provides specialised settings for different shading model.
		*/
		template<STPShadingModel S>
		struct STPShadingModelDescription;

		/**
		 * @brief STPShadingEquation is an adaptive shading model selector for the renderer.
		*/
		class STP_REALISM_API STPShadingEquation : public STPEnvironment::STPSetting {
		private:

			friend class STPScenePipeline;

			/**
			 * @brief Flush the shading model settings to a given program.
			 * @param program The program.
			*/
			virtual void operator()(STPProgramManager&) const = 0;

		public:

			const STPShadingModel Model;

			/**
			 * @brief Initialise a new shading equation instance.
			 * @param model Specifies the shading model being used.
			*/
			STPShadingEquation(STPShadingModel);

			virtual ~STPShadingEquation() = default;

		};

		/* ---------------------------------- Shadow ------------------------------------ */

		/**
		 * @brief STPShadowMapFilterFunction is an adaptive shadow map filter manager for any shadow map filter.
		*/
		class STP_REALISM_API STPShadowMapFilterFunction : public STPEnvironment::STPSetting {
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

			virtual ~STPShadowMapFilterFunction() = default;

			/**
			 * @brief Check if all values for the filter are valid.
			 * @return True if all of them are valid.
			*/
			virtual bool validate() const override;

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
			STPBidirectionalScattering* BSDFObject = nullptr;
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

		const bool hasMaterialLibrary;

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
		 * @return The pointer to the depth shader. Null pointer is returned if depth shader is unused.
		*/
		const STPShaderManager* getDepthShader() const;
		
		/**
		 * @brief Add a light to the scene pipeline.
		 * Exception is thrown if the scene pipeline has no more memory to hold more lights.
		 * @param light The pointer to the light that should be added.
		*/
		void addLight(STPSceneLight&);

		/**
		 * @brief Draw the environment.
		 * @tparam Env An array of environment rendering components.
		 * @param env The pointer the environment object.
		*/
		template<class Env>
		void drawEnvironment(const Env&) const;

		/**
		 * @brief Shade the object.
		 * It assumes the G-Buffer framebuffer is currently bound.
		 * @tparam Clr Set to true to clear the post-process buffer before lighting.
		 * @tparam Ao The ambient occlusion object.
		 * @tparam Pp The post-processing object.
		 * @param ao The pointer to the AO object.
		 * @param post_process The pointer to the post-processing object.
		 * @param mask Specifies the mask for which the pixels should be shaded.
		*/
		template<bool Clr, class Ao, class Pp>
		void shadeObject(const Ao*, const Pp*, unsigned char) const;

	public:

		/**
		 * @brief STPScenePipelineInitialiser contains initialisers and logs from compilations of scene pipeline shaders.
		*/
		struct STPScenePipelineInitialiser {
		public:

			//Defines the maximum memory to be allocated for each array in the shader.
			STPSceneShaderCapacity ShaderCapacity;

			//Specifies the shading model used during light.
			const STPShadingEquation* ShadingModel;
			//Specifies shadow map filter function to be used in the scene.
			const STPShadowMapFilterFunction* ShadowFilter;

			//Pointer to lighting shader initialiser
			const STPScreen::STPScreenInitialiser* GeometryBufferInitialiser;

		};

		/**
		 * @brief Initialise an empty scene pipeline.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		 * @param mat_lib The pointer to the material library.
		 * A null pointer can be provided to indicate no user-defined material library is used for this rendering pipeline.
		 * No rendering component that uses a material library is allowed to be added later, however.
		 * Like camera, user is responsible for its lifetime, and the memory must not be reallocated.
		 * Modification to the data within the material library in runtime is allowed.
		 * @param scene_init The pointer to scene initialiser.
		*/
		STPScenePipeline(const STPCamera&, const STPMaterialLibrary*, const STPScenePipelineInitialiser&);

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
		 * Any pending asynchronous operations will be sync automatically by this function before rendering.
		*/
		void traverse();

	};

#define SHADING_MODEL_DEF(MOD) \
template<> struct STP_REALISM_API STPScenePipeline::STPShadingModelDescription<STPScenePipeline::STPShadingModel::MOD> : \
	public STPScenePipeline::STPShadingEquation

	SHADING_MODEL_DEF(BlinnPhong) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPShadingModelDescription();

		~STPShadingModelDescription() = default;

		bool validate() const override;

		//As Blinn-Phong model does not properly handle roughness material,
		//such that it is emulated by a linear equation and calculate the specular power as a linear interpolation of the roughness.

		//Specifies the range of input of the interpolation, from minimum to maximum.
		glm::vec2 RoughnessRange;
		//Specifies the range of output of the interpolation, same as above.
		glm::vec2 ShininessRange;

	};

#undef SHADING_MODEL_DEF

#define SHADOW_MAP_FILTER_DEF(FILT) \
template<> struct STP_REALISM_API STPScenePipeline::STPShadowMapFilterKernel<STPShadowMapFilter::FILT> : \
	public STPScenePipeline::STPShadowMapFilterFunction

	SHADOW_MAP_FILTER_DEF(PCF) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPShadowMapFilterKernel();

		~STPShadowMapFilterKernel() = default;

		bool validate() const override;

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

		bool validate() const override;

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