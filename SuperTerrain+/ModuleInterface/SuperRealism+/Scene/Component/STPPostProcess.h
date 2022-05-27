#pragma once
#ifndef _STP_POST_PROCESS_H_
#define _STP_POST_PROCESS_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Screen Renderer
#include "STPScreen.h"
//GL Object
#include "../../Object/STPTexture.h"
#include "../../Object/STPSampler.h"

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPPostProcess captures rendering buffer, process the buffer and display the final image.
	*/
	class STP_REALISM_API STPPostProcess : private STPScreen {
	public:

		/**
		 * @brief STPToneMappingFunction selects the function used for applying tone mapping to the final image.
		 * The engine supports some of the state-of-the-art filmic tone mapping functions.
		*/
		enum class STPToneMappingFunction : unsigned char {
			//No tone mapping function will be applied.
			Disable = 0x00u,
			//Introduced by Hajime Uchimura in 2017.
			GranTurismo = 0x01u,
			//Introduced by Timothy Lottes in 2016.
			Lottes = 0x02u,
			//Introduced by John Hable in 2010.
			Uncharted2 = 0x03u
		};

		/**
		 * @brief STPPostEffect specifies type of different post effects supported.
		*/
		enum class STPPostEffect : unsigned char {
			//Gamma is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems.
			Gamma = 0x00u,
			//Contrast is the difference in luminance or colour that makes an object (or its representation in an image or display) distinguishable. 
			Contrast = 0x01u,
			//Exposure is the amount of light per unit area (the image plane illuminance times the exposure time) reaching a frame of photographic film or 
			//the surface of an electronic image sensor, as determined by shutter speed, lens aperture, and scene luminance. 
			Exposure = 0x02u
		};

		/**
		 * @brief STPToneMappingDefinition uses some parameters to define the shape of a particular tone mapping function.
		 * Default parameters loaded in each specialised tone mapping function are the parameters specified by the original author.
		 * Most of the parameter names in each definition are non-intuitive, hence it is recommended to 
		 * visualise the curve in a graph plotter in case you need to tweak the parameters.
		 * @tparam F The tone mapping function used for this curve.
		*/
		template<STPToneMappingFunction F>
		struct STPToneMappingDefinition;

		/**
		 * @brief STPToneMappingCurve is an adaptive tone mapping curve generator for a given tone mapping function.
		*/
		class STP_REALISM_API STPToneMappingCurve {
		private:

			friend class STPPostProcess;

			/**
			 * @brief Generate and update the new curve settings to the program.
			 * @param program The program to the program where curve setting should be updated.
			*/
			virtual void operator()(STPProgramManager&) const = 0;

		public:

			//The type of tone mapping function the tone mapping curve is applied to.
			const STPToneMappingFunction Function;

			/**
			 * @brief Init a tone mapping curve.
			 * @param function The type of the tone mapping function.
			*/
			STPToneMappingCurve(STPToneMappingFunction);

			STPToneMappingCurve(const STPToneMappingCurve&) = default;

			STPToneMappingCurve(STPToneMappingCurve&&) noexcept = default;

			STPToneMappingCurve& operator=(const STPToneMappingCurve&) = delete;

			STPToneMappingCurve& operator=(STPToneMappingCurve&&) = delete;

			virtual ~STPToneMappingCurve() = default;

		};

	private:

		//The post process capturing unit
		STPSampler ImageSampler;

		STPSimpleScreenFrameBuffer PostProcessResultContainer;

	public:

		/**
		 * @brief Init the post processor.
		 * @param tone_mapping A pointer to a base tone mapping curve with specific tone mapping function chosen and parameters loaded.
		 * @param post_process_init The pointer to the post process initialiser.
		*/
		STPPostProcess(const STPToneMappingCurve&, const STPScreenInitialiser&);

		STPPostProcess(const STPPostProcess&) = delete;

		STPPostProcess(STPPostProcess&&) = delete;

		STPPostProcess& operator=(const STPPostProcess&) = delete;

		STPPostProcess& operator=(STPPostProcess&&) = delete;

		~STPPostProcess() = default;

		/**
		 * @brief Get the buffer where all post process memory is stored.
		 * @return A pointer to a texture containing all post process data.
		*/
		const STPTexture& operator*() const;

		/**
		 * @brief Set the value of a particular effect.
		 * @tparap E The type of the effect.
		 * @param val The value to be set to.
		*/
		template<STPPostEffect E>
		void setEffect(float);

		/**
		 * @brief Set the buffer property for post processing.
		 * @param stencil An optional pointer to stencil buffer to be attached to the post process buffer.
		 * @param dimension Specifies the new dimension.
		 * This causes the post process buffer to be reallocated.
		*/
		void setPostProcessBuffer(STPTexture*, glm::uvec2);

		/**
		 * @brief Start rendering to post process buffer.
		 * To stop rendering, bind framebuffer target to any other binding points.
		*/
		void capture() const;

		/**
		 * @brief Render post processed rendering output to the screen.
		*/
		void process() const;

	};

#define TONE_MAPPING_DEF(FUNC) template<> \
	struct STP_REALISM_API STPPostProcess::STPToneMappingDefinition<STPPostProcess::STPToneMappingFunction::FUNC> \
		: public STPPostProcess::STPToneMappingCurve
	
	TONE_MAPPING_DEF(Disable) {
	private:

		inline void operator()(STPProgramManager&) const override {};

	public:

		STPToneMappingDefinition();

		~STPToneMappingDefinition() = default;

	};

	TONE_MAPPING_DEF(GranTurismo) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPToneMappingDefinition();

		~STPToneMappingDefinition() = default;

		//Control the max value of the curve and the contrast of display.
		float MaxBrightness, Contrast;
		//Define the start and length of the linear section.
		float LinearStart, LinearLength;
		//The black fall off, higher value makes dark area more even darker.
		float BlackTightness;
		//Specifies the initial value, i.e., the value when input colour is zero.
		float Pedestal;

	};

	TONE_MAPPING_DEF(Lottes) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPToneMappingDefinition();

		~STPToneMappingDefinition() = default;

		//Control the contrast level of the output image.
		float Contrast;
		//Control the shoulder height.
		float Shoulder;
		//Specifies the max possible brightness the HDR input can reach.
		float HDRMax;
		//The anchors of the curve, specify a point coordinate with respective to x and y where the gradient of curve starts turning.
		glm::vec2 Middle;

	};

	TONE_MAPPING_DEF(Uncharted2) {
	private:

		void operator()(STPProgramManager&) const override;

	public:

		STPToneMappingDefinition();

		~STPToneMappingDefinition() = default;

		//The strength of the shoulder, the top of the curve.
		float ShoulderStrength;
		//The strength of the linear part and the slope of it.
		float LinearStrength, LinearAngle;
		//Control the toe, the bottom of the curve.
		float ToeStrength, ToeNumerator, ToeDenominator;
		//Linear white point value.
		float LinearWhite;

	};

#undef TONE_MAPPING_DEF

}
#endif//_STP_POST_PROCESS_H_