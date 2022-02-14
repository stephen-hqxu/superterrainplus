#pragma once
#ifndef _STP_AMBIENT_OCCLUSION_H_
#define _STP_AMBIENT_OCCLUSION_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Screen
#include "STPScreen.h"
//GL Object
#include "../../Object/STPSampler.h"
#include "../../Object/STPTexture.h"
#include "../../Object/STPBindlessTexture.h"

#include "../../Environment/STPOcclusionKernelSetting.h"
//Output Processing
#include "STPGaussianFilter.h"

//GLM
#include <glm/vec2.hpp>

#include <optional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAmbientOcclusion is an indirect lighting approximation that 
	 * tries to to approximate indirect lighting by darkening creases, holes and surfaces that are closed to each other.
	*/
	class STP_REALISM_API STPAmbientOcclusion : private STPScreen {
	private:

		STPTexture RandomRotationVector;
		std::optional<STPBindlessTexture> RandomRotationVectorHandle;
		//Sampler for the external input
		STPSampler GBufferSampler;

		//ambient occlusion output
		mutable STPSimpleScreenFrameBuffer OcclusionResultContainer;

		//Store the dimension of the texture which contains random rotation vectors
		const glm::uvec2 NoiseDimension;

		STPGaussianFilter BlurWorker;

	public:

		/**
		 * @brief Initialise a new ambient occlusion rendering component.
		 * @param kernel_setting The pointer to the setting to configure how ambient occlusion will be performed.
		 * @param filter A rvalue reference to a Gaussian filter which will be used to blurred the output ambient occlusion.
		 * @param kernel_init The pointer to ambient occlusion initialiser.
		*/
		STPAmbientOcclusion(const STPEnvironment::STPOcclusionKernelSetting&, STPGaussianFilter&&, const STPScreenInitialiser&);

		STPAmbientOcclusion(const STPAmbientOcclusion&) = delete;

		STPAmbientOcclusion(STPAmbientOcclusion&&) = delete;

		STPAmbientOcclusion& operator=(const STPAmbientOcclusion&) = delete;

		STPAmbientOcclusion& operator=(STPAmbientOcclusion&&) = delete;

		~STPAmbientOcclusion() = default;

		/**
		 * @brief Set new screen space dimension and re-initialise ambient occlusion framebuffer.
		 * @param stencil The pointer to the stencil buffer to be used to control ambient occlusion region, or nullptr if not used.
		 * @param dimension The new dimension for the rendering screen.
		*/
		void setScreenSpace(STPTexture*, glm::uvec2);

		/**
		 * @brief Start performing ambient occlusion calculation.
		 * The occlusion result is stored in the internal buffer, this call clears all previous data.
		 * @param depth The texture which contains geometry depth data.
		 * This depth data will be used to reconstruct geometry position.
		 * @param normal The texture which contains geometry normal data.
		 * @param output The pointer to the framebuffer where the final ambient occlusion output will be stored.
		*/
		void occlude(const STPTexture&, const STPTexture&, STPFrameBuffer&) const;

	};

}
#endif//_STP_AMBIENT_OCCLUSION_H_