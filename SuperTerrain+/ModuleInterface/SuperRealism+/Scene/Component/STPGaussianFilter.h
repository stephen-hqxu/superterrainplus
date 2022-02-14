#pragma once
#ifndef _STP_GAUSSIAN_FILTER_H_
#define _STP_GAUSSIAN_FILTER_H_

#include <SuperRealism+/STPRealismDefine.h>
//Off-Screen Rendering
#include "STPScreen.h"
//GL Management
#include "../../Object/STPTexture.h"
#include "../../Object/STPSampler.h"
#include "../../Object/STPFrameBuffer.h"

#include <optional>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPGaussianFilter is a spatial domain non-linear filter in which
	 * each pixel in the resulting image has a value equal to the output from a Gaussian function.
	 * The pixels further away from the filter kernel will have less weight than the pixels closer than it.
	 * This filter currently only supports a single 8-bit channel filtering.
	 * TODO: extend this filter so it support more channel type
	*/
	class STP_REALISM_API STPGaussianFilter : private STPScreen {
	private:

		STPSampler InputImageSampler;
		//Separable filter requires a cache to store intermediate result
		mutable STPSimpleScreenFrameBuffer IntermediateCache;

		glm::vec4 BorderColor;

	public:

		/**
		 * @brief Initialise a Gaussian filter instance.
		 * @param variance Specifies the variance for Gaussian function.
		 * @param sample_distance Specifies the distance between each sampling point on the Gaussian kernel.
		 * @param radius Specifies the radius of the filter kernel.
		 * @param filter_init The pointer to the filter initialiser.
		*/
		STPGaussianFilter(double, double, unsigned int, const STPScreenInitialiser&);

		STPGaussianFilter(const STPGaussianFilter&) = delete;

		STPGaussianFilter(STPGaussianFilter&&) noexcept = default;

		STPGaussianFilter& operator=(const STPGaussianFilter&) = delete;

		STPGaussianFilter& operator=(STPGaussianFilter&&) noexcept = default;

		~STPGaussianFilter() = default;

		/**
		 * @brief Set the dimension of the filter cache.
		 * This causes the program to deallocate all previous memory and reallocate.
		 * @param stencil An optional pointer for framebuffer stencil attachment.
		 * This can be used to control which region to perform the filter on.
		 * @param dimension The new dimension to be set.
		*/
		void setFilterCacheDimension(STPTexture*, glm::uvec2);

		/**
		 * @brief Set the color for pixels outside the main filtering region.
		 * This color affects the filtering region where the convolution kernel reads from neighbouring pixels.
		 * @param border The border color to be set to.
		*/
		void setBorderColor(glm::vec4);

		/**
		 * @brief Perform Gaussian filter on the input texture.
		 * The output will be stored internally and it will remain valid either
		 * until this function is called again or memory reallocation happens or the filter is destroyed.
		 * The input texture must have dimension which is equal or less then the cache dimension,
		 * otherwise this is a undefined behaviour and may or may not terminate the program.
		 * @param input The pointer to the input texture.
		 * @param output Specifies the framebuffer where the filter output will be stored.
		*/
		void filter(const STPTexture&, STPFrameBuffer&) const;

	};

}
#endif//_STP_GAUSSIAN_FILTER_H_