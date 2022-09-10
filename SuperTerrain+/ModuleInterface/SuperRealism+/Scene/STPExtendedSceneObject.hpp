#pragma once
#ifndef _STP_EXTENDED_SCENE_OBJECT_HPP_
#define _STP_EXTENDED_SCENE_OBJECT_HPP_

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
				//- The new geometry from which is traversable handle is built from.
				//- Obviously, this pointer should be accessible from device.
				//- A handle may contain many geometries and instances, each instance should be assigned with an instance ID
				//  using which to locate each instance; each instance contains a lot of vertex data.
				//- Currently only fixed vertex format is supported, first 3 floats are used as position and followed by 2 floats of texture coordinate;
				//  texture coordinate is required to be in range of [0.0f, 1.0f].
				//- A geometry that is not used to build the traversable handle as provided will result in UB.
				const float* const* PrimitiveGeometry;
				//The new geometry index, holds the same requirement as the primitive geometry.
				const uint3* const* PrimitiveIndex;

			};
			/**
			 * @brief Notify the rendering pipeline about an update to the geometry for this traceable object.
			 * The master rendering pipeline will then update the memory with the new information,
			 * this operation takes time so it is fully asynchronous.
			 * The object is responsible for managing the memory for all memories passed.
			 * Until swapBuffer() is called, all data in the memory are not allowed to be changed, doing so will result in race condition;
			 * this function should not be called again before this time either, such notification will be ignored.
			 * @param geometry_info The information regarding the updating geometry.
			*/
			typedef std::function<void(const STPGeometryUpdateInformation&)> STPGeometryUpdateNotifier;

		protected:

			//Call the function to notify the dependent scene pipeline for the geometry update.
			//This function is not thread safe.
			STPGeometryUpdateNotifier NotifyGeometryUpdate;

		public:

			STPObjectID TraceableObjectID = STPExtendedSceneObject::EmptyObjectID;

			/**
			 * @brief Initialise a new traceable object.
			*/
			STPTraceable() = default;

			virtual ~STPTraceable() = default;

			/**
			 * @brief Set the geometry update notifier from the master rendering pipeline.
			 * This function is supposed to be called by the scene pipeline upon object is added to rendering queue.
			 * @param notifier The notifier from the rendering pipeline.
			*/
			void setGeometryUpdateNotifier(STPGeometryUpdateNotifier notifier) noexcept {
				this->NotifyGeometryUpdate = notifier;
			}

			/**
			 * @brief Notify the traceable object to swap buffer.
			 * After swapping, the front buffer should be exactly the same as the memory passed when notifying the rendering pipeline for update.
			 * Of course, violation of that will give you UB.
			 * The previous, old front buffer can be safely recycled by the application.
			*/
			virtual void swapBuffer() = 0;

		};

	}

}
#endif//_STP_EXTENDED_SCENE_OBJECT_HPP_