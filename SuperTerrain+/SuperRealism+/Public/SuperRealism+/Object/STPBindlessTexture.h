#pragma once
#ifndef _STP_BINDLESS_TEXTURE_H_
#define _STP_BINDLESS_TEXTURE_H_

//Dependent GL Object
#include "STPTexture.h"
#include "STPSampler.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPBindlessTexture allows having texture to be shared as an integer number rather than binding to a target.
	 * This is a thin wrapper to bindless texture which requires ARB_bindless_texture extension support on the target GPU platform.
	 * All associated texture and sampler must remain valid until bindless texture is destroyed.
	*/
	namespace STPBindlessTexture {

		//The handle to the bindless texture.
		using STPHandle = STPSmartDeviceObject::STPGLBindlessTextureHandle;

		/**
		 * @brief Create a bindless texture from a managed texture object.
		 * @param texture The pointer to the managed texture object.
		*/
		STP_REALISM_API STPHandle make(const STPTexture& texture) noexcept;

		/**
		 * @brief Create a bindless texture from a managed texture object and sampler object.
		 * @param texture The pointer to the managed texture object.
		 * @param sampler The pointer to the managed sampler object.
		*/
		STP_REALISM_API STPHandle make(const STPTexture& texture, const STPSampler& sampler) noexcept;

		//Definitions are in STPTexture

	}

}
#endif//_STP_BINDLESS_TEXTURE_H_