#pragma once
#ifndef _STP_SCENE_PIPELINE_H_
#define _STP_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Components
#include "./Renderer/STPSun.h"
#include "./Renderer/STPHeightfieldTerrain.h"
//Camera
#include "./Utility/Camera/STPCamera.h"

//GL Object
#include "./Object/STPBuffer.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPScenePipeline is a master rendering pipeline for the terrain engine.
	 * It manages all rendering components and issues rendering task automatically.
	 * It is recommended to have only one scene pipeline for one context.
	*/
	class STP_REALISM_API STPScenePipeline {
	public:

		/**
		 * @brief STPSceneWorkflow defines rendering targets and components for the scene pipeline.
		 * Pointers can be assigned with nothing to indicate no rendering workflow for such target, therefore should be ignored by the pipeline.
		 * All assigned pointers must remain valid until the pipeline is destroied.
		*/
		struct STPSceneWorkflow {
		public:

			//Environment renderer
			STPSun* Sun = nullptr;

			//Terrain renderer
			STPHeightfieldTerrain* Terrain = nullptr;
		};

	private:

		//The master camera for the rendering scene.
		const STPCamera& SceneCamera;
		STPBuffer CameraBuffer;
		void* MappedCameraBuffer;

		//Rendering tasks
		const STPSceneWorkflow Workflow;

		/**
		 * @brief Reset the current framebuffer back to initial state and read for new rendering.
		*/
		void reset() const;

	public:

		/**
		 * @brief Initialise an empty scene pipeline.
		 * @param camera The pointer to the camera.
		 * The camera must remain valid as long as the current scene pipeline is valid.
		 * @param scene The pointer to the scene to be processed and rendered by the pipeline.
		 * The workflow itself is copied by the pipeline, however pipeline stages pointers provided within are 
		 * required to remain valid.
		*/
		STPScenePipeline(const STPCamera&, const STPSceneWorkflow&);

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
		*/
		void traverse() const;

	};

}
#endif//_STP_SCENE_PIPELINE_H_