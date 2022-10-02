#pragma once
#ifndef _STP_EXTENDED_SCENE_PIPELINE_H_
#define _STP_EXTENDED_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>
//GL
#include "./Object/STPTexture.h"
//Object
#include "./Scene/STPExtendedSceneObject.hpp"
//Memory
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>

//OptiX
#include <cuda.h>
#include <optix_types.h>

#include <vector>
#include <set>
#include <ostream>
#include <memory>
#include <type_traits>

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
		 * @brief STPShaderMemoryInternal is the internal component for STPShaderMemory which is opaque to the application.
		*/
		class STPShaderMemoryInternal;

	public:

		/**
		 * @brief STPSceneObjectCapacity controls declared array length for holding scene objects internally.
		 * This limit is set by user and one should not add objects more than this limit.
		*/
		struct STPSceneObjectCapacity {
		public:

			size_t TraceableObject;

		};

		/**
		 * @brief STPShaderMemoryType specifies the type of STPShaderMemory to create.
		 * Different type of memory has different usage, content and size.
		 * Read the documentation of each type carefully.
		*/
		enum class STPShaderMemoryType : unsigned char {
			/**
			 * @brief A stencil buffer to indicate which pixel should have ray launched from.
			 * This stencil buffer also outputs result flags for ray intersection test,
			 * for example which object does the ray hits, or if ray missed everything.
			 * ---
			 * type: input/output
			 * format: R8UI
			 * size: render resolution
			*/
			ScreenSpaceStencil = 0x00u,
			/**
			 * @brief Stores the screen space depth from the rendering scene,
			 * the specific ray origin can be recovered by performing depth reconstruction.
			 * ---
			 * type: input
			 * format: R32F
			 * size: render resolution
			*/
			ScreenSpaceRayDepth = 0x01u,
			/**
			 * @brief Stores screen space ray direction, which specifies a 3-component unit vector towards which ray is launched.
			 * The last component is for padding and unused.
			 * The ray direction should have range in [-1, 1], but due to compatibility reason they should be converted to [0, 1] before storing,
			 * and it will be converted back in the shader.
			 * ---
			 * type: input
			 * format: RGBA16
			 * size: render resolution
			*/
			ScreenSpaceRayDirection = 0x02u,
			/**
			 * @brief Record the 3-component vector of position of intersection in world space.
			 * The last component is for padding hence unused.
			 * ---
			 * type: output
			 * format: RGBA32F
			 * size: render resolution
			*/
			GeometryPosition = 0x03u,
			/**
			 * @brief The 2-component vector of normalised texture coordinate of the intersecting geometry.
			 * Yes, we currently only support normalised UV ([0, 1]).
			 * ---
			 * type: output
			 * format: RG16
			 * size: render resolution
			*/
			GeometryUV = 0x04u,
			
			//The total number of type, it is not a valid type itself.
			TotalTypeCount
		};
		//Value type of STPShaderMemoryType, this raw value can be used as index of each type.
		using STPShaderMemoryType_t = std::underlying_type_t<STPShaderMemoryType>;

		/**
		 * @brief STPShaderMemory is a memory shared between user and the renderer.
		 * User may provide memory input accessible by the shader, and receive output from it.
		*/
		class STP_REALISM_API STPShaderMemory {
		private:

			friend class STPExtendedScenePipeline;

			STPShaderMemoryInternal* Internal;

			//A helper function to access the internal of the memory
			STPShaderMemoryInternal* operator->() const;

		public:

			/**
			 * @brief Initialise a shader memory.
			 * @param internal The internal part of the shader memory.
			 * If not provided, it will be initialised to null pointer.
			*/
			STPShaderMemory(STPShaderMemoryInternal* = nullptr);

			~STPShaderMemory() = default;

			/**
			 * @brief Get the texture object shared with user; this texture object is managed internally by the scene pipeline.
			 * User can modify the content of the texture, but should not trigger reallocation,
			 * doing this will result in undefined behaviour.
			 * 
			 * The rendering pipeline generally don't reallocate the memory unconditionally so user is safe to retain the texture
			 * until either it is released or the scene pipeline is destroyed.
			 * One exception is changing rendering resolution, since most types of shader memory has texture size based on rendering resolution,
			 * such that user should be aware of reallocation when rendering resolution is changed.
			*/
			STPTexture& texture() noexcept;

			/**
			 * @brief Get the type of shader memory.
			 * @return The type of shader memory.
			*/
			STPShaderMemoryType type() const noexcept;

		};

	private:

		/**
		 * @brief STPSceneGraph contains all rendering components for an extended scene pipeline.
		*/
		struct STPSceneGraph {
		public:

			//All traceable object nodes.
			std::vector<STPExtendedSceneObject::STPTraceable*> TraceableObjectDatabase;
			//Record all traversable depths.
			std::multiset<unsigned int> TraceableTraversableDepth;

		};

		/**
		 * @brief STPDeviceContextDestroyer destroys the OptiX device context.
		*/
		struct STPDeviceContextDestroyer {
		public:

			void operator()(OptixDeviceContext) const;

		};
		//For simplicity of management, each instance of scene pipeline holds a context
		const STPUniqueResource<OptixDeviceContext, nullptr, STPDeviceContextDestroyer> Context;

		//The master stream for the ray tracing rendering and all sorts of GPU operations.
		STPSmartDeviceObject::STPStream RendererStream;

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

		//initial render resolution is (0, 0) which allocates no memory for rendering
		glm::uvec2 RenderResolution;

	public:

		/**
		 * @brief Initialiser for the extended scene pipeline.
		*/
		struct STPScenePipelineInitialiser {
		public:

			//Specifies the CUDA device context to be used for creating OptiX context.
			CUcontext DeviceContext;
			//The log level indicates the severity of the message to be returned to the log stream.
			int LogLevel;
			//Specifies if a debug context should be created.
			//This will turn on device validation and all exception handling, which can greatly aid debugging but slow everything down.
			//Only use this for development purposes, remember to turn it off when you are done.
			bool UseDebugContext;
			//Specify the target device architecture for device code generation.
			//The device architecture *<value>* should have the same format as argument supplied to NVRTC compiler "-arch=sm_<value>"
			unsigned int TargetDeviceArchitecture;

			//Defines the maximum memory to be allocated for each array to hold those objects.
			STPSceneObjectCapacity ObjectCapacity;

		};

		/**
		 * @brief STPValidatedInformation is a simple pre-validator for ray tracing launch information.
		 * @tparam Desc The type of information to be validated.
		*/
		template<class Desc>
		class STPValidatedInformation {
		public:

			const Desc Description;

		private:

			friend class STPExtendedScenePipeline;

			/**
			 * @brief Creates a new validated information, the provided descriptor will be validated upon construction.
			 * The descriptor will be copied into the current validator instance and no changes can be made.
			 * However, the memory resides in the underlying pointer should be maintained by the application.
			 * @param descriptor The descriptor object to be validated.
			*/
			STPValidatedInformation(Desc);

		public:

			~STPValidatedInformation() = default;

		};

		/**
		 * @brief STPSSRIDescription provides launch information for tracing screen space ray intersection.
		*/
		struct STP_REALISM_API STPSSRIDescription {
		public:

			STPShaderMemory Stencil, RayDepth, RayDirection, Position, UV;

			/**
			 * @brief Create a validated STPSSRIDescription object.
			 * @return The validated STPSSRIDescription object.
			*/
			STPValidatedInformation<STPSSRIDescription> validate() const;

		};

		/**
		 * @brief Initialise an extended scene pipeline.
		 * @param scene_init The scene initialiser.
		 * @param msg_stream A stream to receive log messages from the device context.
		*/
		STPExtendedScenePipeline(const STPScenePipelineInitialiser&, std::ostream&);

		STPExtendedScenePipeline(const STPExtendedScenePipeline&) = delete;

		STPExtendedScenePipeline(STPExtendedScenePipeline&&) = delete;

		STPExtendedScenePipeline& operator=(const STPExtendedScenePipeline&) = delete;

		STPExtendedScenePipeline& operator=(STPExtendedScenePipeline&&) = delete;

		~STPExtendedScenePipeline();

		/**
		 * @brief Check and update acceleration structure for each scene object.
		 * It should always be called unconditionally since most objects process state changes asynchronously
		 * so AS update may not reflect back to the user immediately;
		 * the renderer will receive notification from each object if an update is ready rather efficiently internally.
		 * User is also recommended to put all related object update code together and call this function way before rendering happens,
		 * to minimise rendering delay and possibly frame freezing due to update.
		*/
		void updateAccelerationStructure();

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

		/**
		 * @brief Set the inverse projection view matrix of the viewer.
		 * @param inv_pv The pointer to the inverse PV matrix.
		*/
		void setInverseProjectionView(const glm::mat4&);

		/**
		 * @brief Create a shader memory.
		 * This shader memory is managed by the current scene pipeline.
		 * @param type Specifies the type of shader memory.
		 * @return The created shader memory with undefined content initially.
		*/
		[[nodiscard]] STPShaderMemory createShaderMemory(STPShaderMemoryType);

		/**
		 * @brief Destroy the shader memory.
		 * User must not access the shader memory after it has been destroyed.
		 * Alternatively the shader memory will be destroyed automatically if the current scene pipeline instance is destroyed.
		 * @param sm The shader memory to be destroyed.
		*/
		void destroyShaderMemory(STPShaderMemory);

		/**
		 * @brief Initiate screen space ray intersection shader to find the closest hit of a ray from the viewer to the scene.
		 * @param stencil_in The stencil buffer input.
		 * This can either be a stencil-only format, or a packed depth-stencil texture.
		 * The stencil will be copied to the internal memory and no change is made.
		 * @param ssri_info Provide a validated SSRI shader memory for additional rendering information.
		*/
		void traceIntersection(const STPTexture&, const STPValidatedInformation<STPSSRIDescription>&);
	
	};

}
#endif//_STP_EXTENDED_SCENE_PIPELINE_H_