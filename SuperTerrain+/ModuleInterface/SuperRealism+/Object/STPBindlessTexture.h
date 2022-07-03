#pragma once
#ifndef _STP_BINDLESS_TEXTURE_H_
#define _STP_BINDLESS_TEXTURE_H_

#include <SuperRealism+/STPRealismDefine.h>
//Dependent GL Object
#include "STPTexture.h"
#include "STPSampler.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPBindlessTexture allows having texture to be shared as an integer number rather than binding to a target.
	 * This is a thin wrapper to bindless texture which requires ARB_bindless_texture extension support on the target GPU platform.
	 * All associated texture and sampler must remain valid until bindless texture is destroyed.
	*/
	class STP_REALISM_API STPBindlessTexture {
	private:

		/**
		 * @brief STPBindlessTextureInvalidater automatically unresident and invalidate the bindless handle.
		*/
		struct STP_REALISM_API STPBindlessTextureInvalidater {
		public:

			void operator()(STPOpenGL::STPuint64) const;

		};
		typedef STPSmartGLuint64Object<STPBindlessTextureInvalidater> STPSmartBindlessTexture;
		//Bindless TBO
		STPSmartBindlessTexture Handle;

		/**
		 * @brief Enable bindless handle so it can be used.
		 * If the bindless handle has been enabled, exception is generated.
		*/
		void enableHandle() const;

	public:

		/**
		 * @brief Create a default empty bindless texture with no handle.
		*/
		STPBindlessTexture() = default;

		/**
		 * @brief Create a bindless texture from a managed texture object.
		 * @param texture The pointer to the managed texture object.
		*/
		STPBindlessTexture(const STPTexture&);

		/**
		 * @brief Create a bindless texture from a raw GL texture buffer object.
		 * @param texture The texture buffer object.
		*/
		STPBindlessTexture(STPOpenGL::STPuint);

		/**
		 * @brief Create a bindless texture from a managed texture object and sampler object.
		 * @param texture The pointer to the managed texture object.
		 * @param sampler The pointer to the managed sampler object.
		*/
		STPBindlessTexture(const STPTexture&, const STPSampler&);

		/**
		 * @brief Create a bindless texture from a raw GL texture buffer object and sampler object.
		 * @param texture The texture buffer object.
		 * @param sampler The sampler object.
		*/
		STPBindlessTexture(STPOpenGL::STPuint, STPOpenGL::STPuint);

		STPBindlessTexture(const STPBindlessTexture&) = delete;

		STPBindlessTexture(STPBindlessTexture&&) noexcept = default;

		STPBindlessTexture& operator=(const STPBindlessTexture&) = delete;

		STPBindlessTexture& operator=(STPBindlessTexture&&) noexcept = default;

		~STPBindlessTexture() = default;

		/**
		 * @brief Get the underlying managed bindless texture handle.
		 * @return The bindless texture handle.
		*/
		STPOpenGL::STPuint64 operator*() const;

		/**
		 * @brief Check if the handle is empty.
		*/
		explicit operator bool() const;

	};

}
#endif//_STP_BINDLESS_TEXTURE_H_