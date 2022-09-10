#pragma once
#ifndef _STP_EXTENDED_SCENE_PIPELINE_H_
#define _STP_EXTENDED_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>
//Object
#include "./Scene/STPExtendedSceneObject.hpp"
//Memory
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GLM
#include <glm/vec2.hpp>

//OptiX
#include <optix_types.h>

#include <memory>
//Container
#include <vector>

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
	private:

		/**
		 * @brief STPSceneGraph contains all rendering components for an extended scene pipeline.
		*/
		struct STPSceneGraph {
		public:

			//All traceable object nodes.
			std::vector<STPExtendedSceneObject::STPTraceable*> TraceableObjectDatabase;
			//All stuff below correspond to each traceable object lives in the scene graph.
			//Such that they all have the same length and same index.
			//And of course, they are all dynamically sized.
			STPSmartDeviceMemory::STPStreamedDeviceMemory<OptixInstance[]> TraceableInstanceDatabase;
			STPSmartDeviceMemory::STPStreamedDeviceMemory<const float* const*[]> TraceablePrimitiveGeometry;
			STPSmartDeviceMemory::STPStreamedDeviceMemory<const uint3* const*[]> TraceablePrimitiveIndex;

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
		//The renderer doesn't need to allocate a lot of memory, so keep release threshold as default zero.
		STPSmartDeviceObject::STPMemPool RendererMemoryPool;

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

		STPExtendedScenePipeline();

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