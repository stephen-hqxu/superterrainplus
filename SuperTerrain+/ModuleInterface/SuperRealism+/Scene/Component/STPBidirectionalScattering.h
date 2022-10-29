#pragma once
#ifndef _STP_BIDIRECTIONAL_SCATTERING_H_
#define _STP_BIDIRECTIONAL_SCATTERING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Screen
#include "STPScreen.h"

#include "../../Environment/STPBidirectionalScatteringSetting.h"

#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPBidirectionalScattering is a compute-oriented rendering component that 
	 * performs bidirectional reflectance and transmission calculations for transparent objects.
	 * It supports rendering of various transparency effect like reflection and refraction.
	*/
	class STP_REALISM_API STPBidirectionalScattering {
	private:

		STPScreen BSDFQuad;

		STPSampler NormalSampler, MaterialSampler;
		//This framebuffer is used for copying data from the scene into the internal memory.
		STPScreen::STPSimpleScreenBindlessFrameBuffer RawSceneColorCopier, RawSceneDepthCopier;

		//Record the dimension of the current buffer.
		glm::uvec2 BufferDimension;

	public:

		/**
		 * @brief Initialise a bidirectional scattering renderer.
		 * @param screen_init The pointer to the off-screen renderer initialiser.
		*/
		STPBidirectionalScattering(const STPScreen::STPScreenInitialiser&);

		STPBidirectionalScattering(const STPBidirectionalScattering&) = delete;

		STPBidirectionalScattering(STPBidirectionalScattering&&) = delete;

		STPBidirectionalScattering& operator=(const STPBidirectionalScattering&) = delete;

		STPBidirectionalScattering& operator=(STPBidirectionalScattering&&) = delete;

		~STPBidirectionalScattering() = default;

		/**
		 * @brief Set the BSDF setting.
		 * @param scattering_setting The pointer to the scattering setting.
		*/
		void setScattering(const STPEnvironment::STPBidirectionalScatteringSetting&);

		/**
		 * @brief Set the dimension of the internal copy buffer.
		 * This should remain the same size as the rendering resolution.
		 * @param dimension The new dimension.
		 * This causes memory reallocation which is expensive, so don't call it unless it is necessary.
		*/
		void setCopyBuffer(glm::uvec2);

		/**
		 * @brief Copy the raw scene content into the internal memory.
		 * These raw scene data should not contain the transparent object.
		 * The current framebuffer binding point will be reset after this function returns.
		 * @param colour The pointer to the colour texture to be copied.
		 * @param depth The pointer to the depth texture to be copied.
		*/
		void copyScene(const STPTexture&, const STPTexture&);

		/**
		 * @brief Perform scattering calculation for the scene using existing dataset.
		 * These data should contain the transparent object to be computed.
		 * @param depth The pointer to the depth texture.
		 * @param normal The pointer to the normal texture.
		 * @param material The pointer to the material texture.
		*/
		void scatter(const STPTexture&, const STPTexture&, const STPTexture&) const;

	};

}
#endif//_STP_BIDIRECTIONAL_SCATTERING_H_