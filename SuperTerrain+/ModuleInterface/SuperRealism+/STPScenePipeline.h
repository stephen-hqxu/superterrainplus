#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Component
#include "./Scene/STPSceneObject.h"
#include "./Scene/STPSceneLight.h"
#include "./Scene/Component/STPPostProcess.h"
#include "./Scene/Component/STPAmbientOcclusion.h"
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
#include <optional>

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
		 * @brief STPShadowMapFilter defines filtering technologies used for post-process shadow maps.
		 * Underlying value greater than VSM denotes VSM-derived shadow map filters.
		*/
		enum class STPShadowMapFilter : unsigned char {
			//Nearest-Neighbour filter, shadow value is read from the nearest pixel.
			Nearest = 0x00u,
			//Bilinear filter, shadow value is read from its neighbours and linearly interpolated.
			Bilinear = 0x01u,
			//Percentage-Closer filter, it attempts to smooth the edge of the shadow using a blur kernel.
			PCF = 0x02u,
			//Stratified Sampled PCF, this is a variant of PCF which uses random stratified sampling when convolving the kernel.
			SSPCF = 0x03u,
			//Variance Shadow Mapping, it uses variance to estimate the likelihood of a pixel that should have shadow 
			//after having the shadow map blurred.
			VSM = 0x10u,
			//Exponential Shadow Mapping, it is a derivation of VSM. Instead of using Chebyshev's inequality to approximate the probability,
			//an exponential function is used.
			ESM = 0x11u
		};

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

			//Special effect nodes
			std::optional<STPAmbientOcclusion> AmbientOcclusionObject;
			std::optional<STPPostProcess> PostProcessObject;

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
		 * @brief Get the shader used for performing additional operations during depth rendering.
		 * @return The poitner to the depth shader. Nullprt is returned if depth shader is unused.
		*/
		const STPShaderManager* getDepthShader() const;

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

			//Log for depth shader compilation
			typedef STPLogStorage<1ull> STPDepthShaderLog;
			STPDepthShaderLog DepthShader;

		};

		/**
		 * @brief Initialise an empty scene pipeline.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		 * @param shader_cap The pointer to a struct that defines the maximum memory to be allocated for each array in the shader.
		 * @param shadow_filter The pointer to the shadow map filter function to be used in the scene.
		 * @param log The pointer to log to output the initial compilation results for scene pipeline.
		*/
		STPScenePipeline(const STPCamera&, const STPSceneShaderCapacity&, const STPShadowMapFilterFunction&, STPScenePipelineLog&);

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
		 * @return The identifier of this light in the scene graph array.
		 * If the light is not registered with the scene, exception is thrown.
		 * This index is valid until the light is removed from the scene, adding new lights won't cause the index to be invalidated.
		*/
		STPLightIdentifier locateLight(const STPSceneLight::STPEnvironmentLight<false>*) const;

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
		 * @param identifier The light identifier that uniquely identifies a light in the scene graph.
		 * Operation is invalid if the index does not correspond to a valid light in the scene.
		 * @param data The data supplied whenever it is applicable for a specific property.
		*/
		template<STPLightPropertyType Prop>
		void setLight(STPLightIdentifier);
		//-----------------------------------
		template<STPLightPropertyType Prop>
		void setLight(STPLightIdentifier, float);
		//----------------------------------------
		void setLight(STPLightIdentifier, const STPEnvironment::STPLightSetting::STPAmbientLightSetting&);
		void setLight(STPLightIdentifier, const STPEnvironment::STPLightSetting::STPDirectionalLightSetting&);

		/**
		 * @brief Traverse the scene graph and render every component in sequential order.
		 * This function does not modify the state of any rendering component.
		 * Any update need to be called by the caller prior to rendering.
		 * Any pending async operations will be sync automatically by this function before rendering.
		*/
		void traverse();

	};

#define SHADOW_MAP_FILTER_DEF(FILT) \
template<> struct STP_REALISM_API STPScenePipeline::STPShadowMapFilterKernel<STPScenePipeline::STPShadowMapFilter::FILT> : public STPScenePipeline::STPShadowMapFilterFunction

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
		//This helps to reduce shadow acen effects when light direction is parallel to the surface.
		float minVariance;

		//Specifies the anisotropy filter level. 
		float AnisotropyFilter;

	};

#undef SHADOW_MAP_FILTER_DEF

}
#include "STPScenePipeline.inl"
#endif//_STP_SCENE_PIPELINE_H_