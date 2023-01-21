#pragma once
#ifndef _STP_SMART_DEVICE_OBJECT_H_
#define _STP_SMART_DEVICE_OBJECT_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "../STPNullablePrimitive.h"

#include <SuperTerrain+/STPOpenGL.h>

//CUDA
#include <cuda_runtime.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPSmartDeviceObject is a collection of CUDA objects with automatic lifetime management.
	*/
	namespace STPSmartDeviceObject {

		//Internal implementation for STPSmartDeviceObject
		namespace STPImplementation {

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

			/**
			 * @brief Unregister CUDA graphics resource.
			*/
			struct STP_API STPGraphicsResourceUnregisterer {
			public:

				void operator()(cudaGraphicsResource_t) const;

			};

			/**
			 * @brief Delete GL texture object.
			*/
			struct STP_API STPGLTextureDeleter {
			public:

				void operator()(STPOpenGL::STPuint) const noexcept;

			};

			/**
			 * @brief Unresident bindless GL texture handle.
			*/
			struct STP_API STPGLTextureHandleUnresidenter {
			public:

				void operator()(STPOpenGL::STPuint64) const noexcept;

			};

		}

		//STPStream is a smartly managed CUDA stream object.
		using STPStream = STPUniqueResource<cudaStream_t, nullptr, STPImplementation::STPStreamDestroyer>;
		//STPEvent is a smartly managed CUDA event object.
		using STPEvent = STPUniqueResource<cudaEvent_t, nullptr, STPImplementation::STPEventDestroyer>;
		//STPMemPool is a smartly managed CUDA memory pool object.
		using STPMemPool = STPUniqueResource<cudaMemPool_t, nullptr, STPImplementation::STPMemPoolDestroyer>;
		//STPTexture is a smartly managed CUDA texture object.
		using STPTexture = STPUniqueResource<cudaTextureObject_t, 0ull, STPImplementation::STPTextureDestroyer>;
		//STPSurface is a smartly managed CUDA surface object.
		using STPSurface = STPUniqueResource<cudaSurfaceObject_t, 0ull, STPImplementation::STPSurfaceDestroyer>;
		//STPGraphicsResource is a smartly managed CUDA graphics resource.
		using STPGraphicsResource = STPUniqueResource<cudaGraphicsResource_t, nullptr, STPImplementation::STPGraphicsResourceUnregisterer>;

		//STPGLTextureObject is a smartly managed GL texture object.
		using STPGLTextureObject = STPUniqueResource<STPOpenGL::STPuint, 0u, STPImplementation::STPGLTextureDeleter>;
		//STPGLBindlessTextureHandle is a smartly managed GL bindless texture handle to texture object.
		using STPGLBindlessTextureHandle = STPUniqueResource<STPOpenGL::STPuint64, 0ull, STPImplementation::STPGLTextureHandleUnresidenter>;

		/**
		 * @brief Create a new CUDA stream.
		 * @param flag Specifies the flag for the created stream.
		 * @return A managed CUDA stream object.
		*/
		STP_API STPStream makeStream(unsigned int);

		/**
		 * @brief Create a new CUDA stream with priority.
		 * @param flag Specifies the flag for the created stream.
		 * @param priority Specifies the priority for the stream.
		 * @return A managed CUDA stream object.
		*/
		STP_API STPStream makeStream(unsigned int, int);

		/**
		 * @brief Create a new CUDA event.
		 * @param flag Specifies the flag for the created event.
		 * @return A managed CUDA event object.
		*/
		STP_API STPEvent makeEvent(unsigned int);

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

		/**
		 * @brief Create a new CUDA graphics resource from GL buffer object.
		 * @param buffer Name of the buffer to be registered.
		 * @param flags Register flags.
		 * @return A managed CUDA graphics resource object.
		*/
		STP_API STPGraphicsResource makeGLBufferResource(STPOpenGL::STPuint, unsigned int);

		/**
		 * @brief Create a new CUDA graphics resource from GL texture or renderbuffer object.
		 * @param image Name of the texture of renderbuffer object to be registered.
		 * @param target Identifies the type of object specified by `image`.
		 * @param flags Register flags.
		 * @return A managed CUDA graphics resource object.
		*/
		STP_API STPGraphicsResource makeGLImageResource(STPOpenGL::STPuint, STPOpenGL::STPenum, unsigned int);

		/**
		 * @brief Create GL texture objects.
		 * @param target Specifies the effective texture target of each created texture.
		 * @return A managed GL texture object.
		*/
		STP_API STPGLTextureObject makeGLTextureObject(STPOpenGL::STPenum) noexcept;

		/**
		 * @brief Create a texture handle using the current state of the texture, including any embedded sampler state.
		 * @param texture The texture where bindless handle is created from.
		 * @return A managed GL bindless texture handle.
		*/
		STP_API STPGLBindlessTextureHandle makeGLBindlessTextureHandle(STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Create a texture handle using the current state of the texture, and an external sampler state.
		 * @param texture The texture where bindless handle is created from.
		 * @param sampler The sampler where sampler states are fetched.
		 * @return A managed GL bindless texture handle.
		*/
		STP_API STPGLBindlessTextureHandle makeGLBindlessTextureHandle(STPOpenGL::STPuint, STPOpenGL::STPuint) noexcept;

	}

}
#endif//_STP_SMART_DEVICE_OBJECT_H_