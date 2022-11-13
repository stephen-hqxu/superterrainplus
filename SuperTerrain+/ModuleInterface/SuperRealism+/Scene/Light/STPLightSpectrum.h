#pragma once
#ifndef _STP_LIGHT_SPECTRUM_H_
#define _STP_LIGHT_SPECTRUM_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../../Object/STPBindlessTexture.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLightSpectrum allows generating a light spectrum for looking up colour for light.
	*/
	class STP_REALISM_API STPLightSpectrum {
	private:

		//The generated spectrum, it is a 1D texture of light colour
		STPTexture Spectrum;
		STPBindlessTexture::STPHandle SpectrumHandle;

	public:

		//Record the length of the spectrum, i.e., the number of pixel.
		const unsigned int SpectrumLength;

		/**
		 * @brief Init a light spectrum object.
		 * @param length The length of the light spectrum.
		 * @param format Specify the sized channel format for the spectrum.
		*/
		STPLightSpectrum(unsigned int, STPOpenGL::STPenum);

		STPLightSpectrum(const STPLightSpectrum&) = delete;

		STPLightSpectrum(STPLightSpectrum&&) noexcept = default;

		STPLightSpectrum& operator=(const STPLightSpectrum&) = delete;

		STPLightSpectrum& operator=(STPLightSpectrum&&) noexcept = default;

		virtual ~STPLightSpectrum() = default;

		/**
		 * @brief Get the light spectrum.
		 * @return The pointer to a GL 1D texture of a light spectrum.
		*/
		const STPTexture& spectrum() const noexcept;

		/**
		 * @brief Get the light spectrum handle.
		 * @return The texture handle to the light spectrum.
		*/
		STPOpenGL::STPuint64 spectrumHandle() const noexcept;

		/**
		 * @brief Set the light spectrum with new array of colours.
		 * @param size The number of element in the array.
		 * @param format The pixel format.
		 * @param type The data type of pixel.
		 * @param data An array of data.
		*/
		void setData(STPOpenGL::STPsizei, STPOpenGL::STPenum, STPOpenGL::STPenum, const void*) noexcept;

	};

}
#endif//_STP_LIGHT_SPECTRUM_H_