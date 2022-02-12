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
#include "../../Object/STPFrameBuffer.h"
#include "../../Object/STPProgramManager.h"

#include "../../Environment/STPOcclusionKernelSetting.h"

//GLM
#include <glm/vec2.hpp>

#include <optional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAmbientOcclusion is an indirect lighting approximation that 
	 * tries to to approximate indirect lighting by darkening creases, holes and surfaces that are closed to each other.
	*/
	class STP_REALISM_API STPAmbientOcclusion : public STPScreen {
	private:

		STPTexture RandomRotationVector;
		std::optional<STPBindlessTexture> RandomRotationVectorHandle;
		//Sampler for the external input
		STPSampler GBufferSampler;

		//ambient occlusion output
		STPTexture OcclusionResult;
		STPFrameBuffer OcclusionContainer;

		STPProgramManager OcclusionCalculator;

		//Store the dimension of the texture which contains random rotation vectors
		const glm::uvec2 NoiseDimension;

	public:

		struct STPAmbientOcclusionLog {
		public:

			STPScreenLog QuadShader;
			STPLogStorage<2ull> AOShader;

		};

		/**
		 * @brief Initialise a new ambient occlusion rendering component.
		 * @param kernel_setting The pointer to the setting to configure how ambient occlusion will be performed.
		 * @param log The pointer to log to hold the shader compilation output.
		*/
		STPAmbientOcclusion(const STPEnvironment::STPOcclusionKernelSetting&, STPAmbientOcclusionLog&);

		STPAmbientOcclusion(const STPAmbientOcclusion&) = delete;

		STPAmbientOcclusion(STPAmbientOcclusion&&) = delete;

		STPAmbientOcclusion& operator=(const STPAmbientOcclusion&) = delete;

		STPAmbientOcclusion& operator=(STPAmbientOcclusion&&) = delete;

		~STPAmbientOcclusion() = default;

		/**
		 * @brief Get the texture where the occlusion result is stored to.
		 * @return The pointer to the texture with the occlusion result.
		 * Note that it is not recommended to create a bindless handle from this texture 
		 * since the texture is subject to change.
		*/
		const STPTexture& operator*() const;

		/**
		 * @brief Set the size of the rendering screen.
		 * This function changes how the screen-space buffer is sampled and triggers reallocation to the internal buffer,
		 * therefore it is considered to be expensive.
		 * @param dimension The new dimension for the rendering screen.
		*/
		void setScreenSpaceDimension(glm::uvec2);

		/**
		 * @brief Start performing ambient occlusion calculation.
		 * The occlusion result is stored in the internal buffer, this call clears all previous data.
		 * The framebuffer state is changed by this function.
		 * @param depth The texture which contains geometry depth data.
		 * This depth data will be used to reconstruct geometry position.
		 * @param normal The texture which contains geometry normal data.
		*/
		void occlude(const STPTexture&, const STPTexture&) const;

	};

}
#endif//_STP_AMBIENT_OCCLUSION_H_