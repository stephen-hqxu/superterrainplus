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

//GLM
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPGaussianFilter is a spatial domain non-linear filter in which
	 * each pixel in the resulting image has a value equal to the output from a Gaussian function.
	 * The pixels further away from the filter kernel will have less weight than the pixels closer than it.
	 * This filter currently only supports a single 8-bit channel filtering.
	*/
	class STP_REALISM_API STPGaussianFilter {
	public:

		/**
		 * @brief STPFilterVariant specifies the variant of Gaussian filter.
		*/
		enum class STPFilterVariant : unsigned char {
			//Classic Gaussian filter which assigns a Gaussian weight to each pixel in the kernel.
			GaussianFilter = 0x00u,
			//Bilateral filter weight both the pixel value and the distance from the centre.
			//For our application, bilateral filter is "depth aware", meaning it avoids filter across edges.
			BilateralFilter = 0x01u
		};

		/**
		 * @brief STPFilterKernel defines a specialisation of Gaussian filter variant.
		*/
		template<STPFilterVariant FV>
		struct STPFilterKernel;

		/**
		 * @brief STPKernelExecution runs the filter based on the kernel definition.
		*/
		class STP_REALISM_API STPFilterExecution {
		private:

			friend class STPGaussianFilter;

			/**
			 * @brief Send uniform data to the program.
			 * @param program The program where data is sent.
			*/
			virtual void operator()(STPProgramManager&) const;

		public:

			const STPFilterVariant Variant;

			//Specifies the variance for Gaussian function.
			double Variance;
			//Specifies the distance between each sampling point on the Gaussian kernel.
			double SampleDistance;
			//Specifies the radius of the filter kernel.
			unsigned int Radius;

			/**
			 * @brief Initialise a new filter execution.
			 * @param variant The variant of Gaussian filter.
			*/
			STPFilterExecution(STPFilterVariant);

			virtual ~STPFilterExecution() = default;

		};

	private:

		STPScreen GaussianQuad;

		STPSampler InputImageSampler, InputDepthSampler;
		//Separable filter requires a cache to store intermediate result
		mutable STPScreen::STPSimpleScreenFrameBuffer IntermediateCache;

		glm::vec4 BorderColor;

	public:

		/**
		 * @brief Initialise a Gaussian filter instance.
		 * @param execution Specifies the filter kernel variant.
		 * @param filter_init The pointer to the filter initialiser.
		*/
		STPGaussianFilter(const STPFilterExecution&, const STPScreen::STPScreenInitialiser&);

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
		 * @brief Set the colour for pixels outside the main filtering region.
		 * This colour affects the filtering region where the convolution kernel reads from neighbouring pixels.
		 * @param border The border colour to be set to.
		*/
		void setBorderColor(glm::vec4);

		/**
		 * @brief Perform Gaussian filter on the input texture.
		 * The output will be stored internally and it will remain valid either
		 * until this function is called again or memory reallocation happens or the filter is destroyed.
		 * The input texture must have dimension which is equal or less then the cache dimension,
		 * otherwise this is a undefined behaviour and may or may not terminate the program.
		 * @param depth The depth texture.
		 * Not all filter variants use depth information.
		 * @param input The pointer to the input texture.
		 * @param output Specifies the framebuffer where the filter output will be stored.
		 * @param output_blending True to enable blending for the output.
		 * Blend function is not modified.
		*/
		void filter(const STPTexture&, const STPTexture&, STPFrameBuffer&, bool) const;

	};

#define FILTER_KERNEL(VAR) template<> \
	struct STP_REALISM_API STPGaussianFilter::STPFilterKernel<STPGaussianFilter::STPFilterVariant::VAR> \
		: public STPGaussianFilter::STPFilterExecution

	FILTER_KERNEL(GaussianFilter) {
	public:

		STPFilterKernel();

		~STPFilterKernel() = default;

	};

	FILTER_KERNEL(BilateralFilter) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPFilterKernel();

		~STPFilterKernel() = default;

		//Specifies how much should filter preserve the edge.
		float Sharpness;

	};

#undef FILTER_KERNEL

}
#endif//_STP_GAUSSIAN_FILTER_H_