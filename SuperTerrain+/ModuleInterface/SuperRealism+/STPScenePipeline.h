#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Components
#include "./Renderer/STPSun.h"
#include "./Renderer/STPHeightfieldTerrain.h"
#include "./Renderer/STPPostProcess.h"
//Camera
#include "./Utility/Camera/STPCamera.h"

//GL Object
#include "./Object/STPBuffer.h"

//GLM
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPScenePipeline is a master rendering pipeline for the terrain engine.
	 * It manages all rendering components and issues rendering task automatically.
	 * It is recommended to have only one scene pipeline for one context.
	 * To allow rendering with the scene pipeline, a valid GL function header needs to be included before including this header.
	*/
	class STP_REALISM_API STPScenePipeline : private STPCamera::STPStatusChangeCallback {
	public:

		/**
		 * @brief STPSceneWorkflow defines rendering targets and components for the scene pipeline.
		 * All assigned pointers must remain valid until the pipeline is destroied.
		*/
		struct STPSceneWorkflow {
		public:

			//Environment renderer
			STPSun* Sun = nullptr;
			//Terrain renderer
			const STPHeightfieldTerrain<false>* Terrain = nullptr;
			//Post processing
			STPPostProcess* PostProcess = nullptr;

		};

		/**
		 * @brief STPRenderComponent defines components that need to be processed by the scene pipeline.
		 * Each component is a bitfield flag.
		*/
		typedef unsigned short STPRenderComponent;

		//Render nothing
		constexpr static STPRenderComponent RenderComponentNone = 0u;
		//Enable sun rendering
		constexpr static STPRenderComponent RenderComponentSun = 1u << 0u;
		//Enable terrain rendering
		constexpr static STPRenderComponent RenderComponentTerrain = 1u << 1u;
		//Enable post processing
		constexpr static STPRenderComponent RenderComponentPostProcess = 1u << 2u;

	private:

		//The master camera for the rendering scene.
		const STPCamera& SceneCamera;
		mutable STPBuffer CameraBuffer;
		void* MappedCameraBuffer;

		//some flags to indicate buffer update status
		mutable bool updatePosition, updateView, updateProjection;

		/**
		 * @brief Update data in scene pipeline buffer.
		*/
		void updateBuffer() const;

		void onMove(const STPCamera&) override;

		void onRotate(const STPCamera&) override;

		void onReshape(const STPCamera&) override;

		/**
		 * @brief Query the viewport information in the current context.
		 * @return The X, Y coordinate and the width, height of the viewport.
		*/
		glm::ivec4 getViewport() const;

	public:

		/**
		 * @brief Initialise an empty scene pipeline.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		*/
		STPScenePipeline(const STPCamera&);

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
		 * @brief Traverse the scene graph and render every component in sequential order.
		 * This function does not modify the state of any rendering component, such as view position change.
		 * Any update need to be called by the caller prior to rendering.
		 * Any pending async operations will be sync automatically by this function before rendering.
		 * @tparam R A bit flag field indicating which components are used for rendering.
		 * @param workflow A pointer to scene workflow to process each defined rendering component.
		 * If the bit field indicates some components are unused, the corresponding component can be nullptr.
		 * Otherwise it must be a valid pointer and should remain valid until this function returns.
		*/
		template<STPRenderComponent R>
		void traverse(const STPSceneWorkflow&) const;

	};

}
#include "STPScenePipeline.inl"
#endif//_STP_SCENE_PIPELINE_H_