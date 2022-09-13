#pragma once
#ifndef _STP_EXTENDED_SCENE_PIPELINE_H_
#define _STP_EXTENDED_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>
//Object
#include "./Scene/STPExtendedSceneObject.hpp"
//Memory
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GLM
#include <glm/vec2.hpp>

//OptiX
#include <optix_types.h>

#include <vector>
#include <memory>
#include <mutex>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPExtendedScenePipeline is an extension on STPScenePipeline.
	 * The traditional rendering pipeline provides basic rendering functionality with rasterisation.
	 * Due to limited ability of such rendering technique when comes to ultimate visual quality,
	 * the extended rendering pipeline introduces global illumination.
	 * This new scene pipeline allows combination of the vanilla scene pipeline to achieve hybrid rendering.
	 * Similar to STPScenePipeline, it is recommended to create only one STPExtendedScenePipeline instance per program.
	 * @see STPScenePipeline
	*/
	class STP_REALISM_API STPExtendedScenePipeline {
	public:

		/**
		 * @brief STPSceneObjectCapacity controls declared array length for holding scene objects internally.
		 * This limit is set by user and one should not add objects more than this limit.
		*/
		struct STPSceneObjectCapacity {
		public:

			size_t TraceableObject;

		};

	private:

		/**
		 * @brief STPSceneGraph contains all rendering components for an extended scene pipeline.
		*/
		struct STPSceneGraph {
		public:

			//All traceable object nodes.
			std::vector<STPExtendedSceneObject::STPTraceable*> TraceableObjectDatabase;

		};

		/**
		 * @brief STPDeviceContextDestroyer destroys the OptiX device context.
		*/
		struct STPDeviceContextDestroyer {
		public:

			void operator()(OptixDeviceContext) const;

		};
		//For simplicity of management, each instance of scene pipeline holds a context
		STPUniqueResource<OptixDeviceContext, nullptr, STPDeviceContextDestroyer> Context;

		//The master stream for the ray tracing rendering and all sorts of GPU operations.
		STPSmartDeviceObject::STPStream RendererStream;
		//An event placed at the end of the main rendering command.
		STPSmartDeviceObject::STPEvent RendererEvent;
		//A lock to ensure memory used by renderer is not currently updated, and vice versa.
		//This should be used together with renderer event to ensure both host and device side synchronisation.
		mutable std::mutex RendererMemoryLock;

		STPSceneObjectCapacity SceneMemoryCurrent;
		const STPSceneObjectCapacity SceneMemoryLimit;

		STPSceneGraph SceneComponent;

		/**
		 * @brief Manages memory for the rendering pipeline, including acceleration structure, geometry and texture.
		 * It also handles asynchronous AS build queries from each object in the scene graph.
		*/
		class STPMemoryManager;
		std::unique_ptr<STPMemoryManager> SceneMemory;
		/**
		 * @brief A simple utility that launches a ray from screen space and reports intersection.
		 * It also provides geometry data where the ray intersects, this is useful for rendering mirror reflection.
		 * The ray terminates at the closest hit, or missed.
		*/
		class STPScreenSpaceRayIntersection;
		std::unique_ptr<STPScreenSpaceRayIntersection> IntersectionTracer;

	public:

		/**
		 * @brief Initialiser for the extended scene pipeline.
		*/
		struct STPScenePipelineInitialiser {
		public:

			//Defines the maximum memory to be allocated for each array to hold those objects.
			STPSceneObjectCapacity ObjectCapacity;
			//Specify the target device architecture for device code generation.
			//The device architecture *<value>* should have the same format as argument supplied to NVRTC compiler "-arch=sm_<value>"
			unsigned int TargetDeviceArchitecture;

		};

		STPExtendedScenePipeline(const STPScenePipelineInitialiser&);

		STPExtendedScenePipeline(const STPExtendedScenePipeline&) = delete;

		STPExtendedScenePipeline(STPExtendedScenePipeline&&) = delete;

		STPExtendedScenePipeline& operator=(const STPExtendedScenePipeline&) = delete;

		STPExtendedScenePipeline& operator=(STPExtendedScenePipeline&&) = delete;

		~STPExtendedScenePipeline();

		/**
		 * @brief Add an extended scene object to the extended scene pipeline.
		 * @param object The pointer to extended object to be added.
		 * The pointer is retained by the rendering pipeline, user is responsible for managing the memory.
		*/
		void add(STPExtendedSceneObject::STPTraceable&);

		/**
		 * @brief Set the rendering resolution.
		 * This will cause a plenty of memory reallocation, use with caution because it is slow.
		 * It is recommended to use the same resolution as the rasterisation rendering to avoid artefacts.
		 * @param resolution The new resolution.
		*/
		void setResolution(glm::uvec2);
	
	};

}
#endif//_STP_EXTENDED_SCENE_PIPELINE_H_