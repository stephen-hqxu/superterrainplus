#pragma once
#ifndef _STP_SMART_DEVICE_OBJECT_H_
#define _STP_SMART_DEVICE_OBJECT_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "../STPNullablePrimitive.h"

//CUDA
#include <cuda_runtime.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPSmartDeviceObject is a collection of CUDA objects with automatic lifetime management.
	*/
	namespace STPSmartDeviceObject {

		//Internal implementation for STPSmartDeviceObject
		namespace STPSmartDeviceObjectImpl {

			/**
			 * @brief Destroys CUDA stream.
			*/
			struct STP_API STPStreamDestroyer {
			public:

				void operator()(cudaStream_t) const;

			};

			/**
			 * @brief Destroys CUDA event.
			*/
			struct STP_API STPEventDestroyer {
			public:

				void operator()(cudaEvent_t) const;

			};

			/**
			 * @brief Destroys CUDA memory pool.
			*/
			struct STP_API STPMemPoolDestroyer {
			public:

				void operator()(cudaMemPool_t) const;

			};

			/**
			 * @brief Destroys CUDA texture object.
			*/
			struct STP_API STPTextureDestroyer {
			public:

				void operator()(cudaTextureObject_t) const;

			};

			/**
			 * @brief Destroys CUDA surface object.
			*/
			struct STP_API STPSurfaceDestroyer {
			public:

				void operator()(cudaSurfaceObject_t) const;

			};

		}

		//STPStream is a smartly managed CUDA stream object.
		using STPStream = STPUniqueResource<cudaStream_t, nullptr, STPSmartDeviceObjectImpl::STPStreamDestroyer>;
		//STPEvent is a smartly managed CUDA event object.
		using STPEvent = STPUniqueResource<cudaEvent_t, nullptr, STPSmartDeviceObjectImpl::STPEventDestroyer>;
		//STPMemPool is a smartly managed CUDA memory pool object.
		using STPMemPool = STPUniqueResource<cudaMemPool_t, nullptr, STPSmartDeviceObjectImpl::STPMemPoolDestroyer>;
		//STPTexture is a smartly managed CUDA texture object.
		using STPTexture = STPUniqueResource<cudaTextureObject_t, 0ull, STPSmartDeviceObjectImpl::STPTextureDestroyer>;
		//STPSurface is a smartly managed CUDA surface object.
		using STPSurface = STPUniqueResource<cudaSurfaceObject_t, 0ull, STPSmartDeviceObjectImpl::STPSurfaceDestroyer>;

		/**
		 * @brief Create a new CUDA stream.
		 * @param flag Specifies the flag for the created stream; default is cudaStreamDefault.
		 * @return A managed CUDA stream object.
		*/
		STP_API STPStream makeStream(unsigned int = cudaStreamDefault);

		/**
		 * @brief Create a new CUDA stream with priority.
		 * @param flag Specifies the flag for the created stream.
		 * @param priority Specifies the priority for the stream.
		 * @return A managed CUDA stream object.
		*/
		STP_API STPStream makeStream(unsigned int, int);

		/**
		 * @brief Create a new CUDA event.
		 * @param flag Specifies the flag for the created event; default is cudaEventDefault.
		 * @return A managed CUDA event object.
		*/
		STP_API STPEvent makeEvent(unsigned int = cudaEventDefault);

		/**
		 * @brief Create a new CUDA memory pool object.
		 * @param props A CUDA memory pool properties.
		 * @return A managed CUDA memory pool object.
		*/
		STP_API STPMemPool makeMemPool(const cudaMemPoolProps&);

		/**
		 * @brief Create a new CUDA texture object.
		 * @param resource_desc A CUDA resource description.
		 * @param texture_desc A CUDA texture description.
		 * @param resview_des A CUDA resource view description; this is optional.
		 * @return A managed CUDA texture object.
		*/
		STP_API STPTexture makeTexture(const cudaResourceDesc&, const cudaTextureDesc&, const cudaResourceViewDesc* = nullptr);

		/**
		 * @brief Create a new CUDA surface object.
		 * @param resource_desc A CUDA resource description.
		 * @return A managed CUDA surface object.
		*/
		STP_API STPSurface makeSurface(const cudaResourceDesc&);

	}

}
#endif//_STP_SMART_DEVICE_OBJECT_H_