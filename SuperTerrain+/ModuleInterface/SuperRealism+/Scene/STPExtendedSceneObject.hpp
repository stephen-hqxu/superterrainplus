#pragma once
#ifndef _STP_EXTENDED_SCENE_OBJECT_HPP_
#define _STP_EXTENDED_SCENE_OBJECT_HPP_

#include <SuperTerrain+/Utility/STPThreadPool.h>

//OptiX
#include <optix_types.h>
#include <vector_types.h>

//System
#include <functional>
#include <limits>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPExtendedSceneObject is an extension from STPSceneObject,
	 * and is a collection of all different renderable objects in an extended scene.
	 * @see STPSceneObject
	*/
	namespace STPExtendedSceneObject {

		//An object ID can be used to uniquely identify an extended scene object in the scene pipeline.
		//This ID will be set by the scene pipeline when the current instance is added to that.
		typedef unsigned int STPObjectID;
		//The initial object ID, which is invalid.
		constexpr static STPObjectID EmptyObjectID = std::numeric_limits<STPObjectID>::max();

		/**
		 * @brief STPTraceable is a base class for an object that can be rendered via ray tracing.
		*/
		class STPTraceable {
		public:

			/**
			 * @brief STPGeometryUpdateInformation holds data to acknowledge the rendering pipeline with geometry update.
			 * @see STPGeometryUpdateNotifier
			*/
			struct STPGeometryUpdateInformation {
			public:

				//The pointer to the current traceable object where the function is called.
				//A pointer to any other traceable object will result in UB.
				STPTraceable* SourceTraceable;
				//The new IAS containing a traversable handle pointing to the newly updated child AS.
				OptixInstance Instance;
				/**
				 * @brief - The new geometry from which is traversable handle is built from.
				 * - Obviously, this pointer should be accessible from device.
				 * - A handle may contain many geometries and instances, each instance should be assigned with an instance ID
				 *   using which to locate each instance; each instance contains a lot of vertex data.
				 * - Currently only fixed vertex format is supported, first 3 floats are used as position and followed by 2 floats of texture coordinate;
				 *   texture coordinate is required to be in range of [0.0f, 1.0f].
				 * - A geometry that is not used to build the traversable handle as provided will result in UB.
				*/
				const float* const* PrimitiveGeometry;
				//The new geometry index, holds the same requirement as the primitive geometry.
				const uint3* const* PrimitiveIndex;

			};
			/**
			 * @brief Notify the rendering pipeline about an update to the geometry for this traceable object.
			 * The master rendering pipeline will then update the memory with the new information,
			 * this operation takes time so it is fully asynchronous thus it is safe to be called from a thread other than the main rendering thread.
			 * The object is responsible for managing the memory for all memories passed.
			 * Until swapBuffer() is called, all data in the memory are not allowed to be changed, doing so will result in race condition;
			 * this function should not be called again before this time either, such notification will be ignored.
			 * @param geometry_info The information regarding the updating geometry.
			*/
			typedef std::function<void(const STPGeometryUpdateInformation&)> STPGeometryUpdateNotifier;

			/**
			 * @brief Information from the rendering pipeline.
			*/
			struct STPSceneInformation {
			public:

				//The device context from the master scene pipeline.
				OptixDeviceContext DeviceContext = nullptr;

				/**
				 * @brief This thread pool is shared with the rendering pipeline.
				 * The renderer holds the ownership of this, so it guarantees all pending works are finished before it is destroyed.
				 * This thread pool is intended to be used by the traceable object to issue asynchronous GAS build command.
				 * The thread pool is null if the current object is not attached to any valid scene pipeline.
				*/
				STPThreadPool* GeometryUpdateWorker = nullptr;
				//Call the function to notify the dependent scene pipeline for the geometry update.
				//This function is thread safe.
				STPGeometryUpdateNotifier NotifyGeometryUpdate;

			};

		protected:

			STPSceneInformation SceneInformation;

		public:

			STPObjectID TraceableObjectID = STPExtendedSceneObject::EmptyObjectID;

			/**
			 * @brief Initialise a new traceable object.
			*/
			STPTraceable() = default;

			virtual ~STPTraceable() = default;

			/**
			 * @brief Set the information regarding the rendering pipeline.
			 * This function is intended to be called by the scene pipeline upon object is added to the scene graph.
			 * @param scene_info The information about the scene pipeline.
			*/
			void setSceneInformation(STPSceneInformation scene_info) noexcept {
				this->SceneInformation = scene_info;
			}

			/**
			 * @brief Notify the traceable object to swap buffer.
			 * After swapping, the front buffer should be exactly the same as the memory passed when notifying the rendering pipeline for update.
			 * Of course, violation of that will give you UB.
			 * The previous, old front buffer can be safely recycled by the application.
			*/
			virtual void swapBuffer() noexcept = 0;

			/**
			 * @brief Get the depth of the traversable graph used by this object.
			 * This function should return a compile-time constant value, otherwise UB.
			 * If there exists multiple structure of traversable graph, pick the largest depth among all of them.
			 * @return The traversable graph depth used by this object.
			*/
			virtual unsigned int traversableDepth() noexcept = 0;

		};

	}

}
#endif//_STP_EXTENDED_SCENE_OBJECT_HPP_