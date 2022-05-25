#pragma once
#ifndef _STP_LIGHT_SPECTRUM_H_
#define _STP_LIGHT_SPECTRUM_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../../Object/STPBindlessTexture.h"

//GLM
#include <glm/vec3.hpp>

//System
#include <vector>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLightSpectrum allows generating a light spectrum for looking up colour for light.
	*/
	class STP_REALISM_API STPLightSpectrum {
	private:

		//The generated spectrum, it is a 1D texture of light colour
		STPTexture Spectrum;
		STPBindlessTexture SpectrumHandle;

	public:

		//Contains an array of colour
		typedef std::vector<glm::vec3> STPColorArray;

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
		const STPTexture& spectrum() const;

		/**
		 * @brief Get the light spectrum handle.
		 * @return The texture handle to the light spectrum.
		*/
		STPOpenGL::STPuint64 spectrumHandle() const;

		/**
		 * @brief Set the light spectrum with new array of colours.
		 * @param colour The array of colour to be filled into the spectrum.
		*/
		void setData(const STPColorArray&);

	};

}
#endif//_STP_LIGHT_SPECTRUM_H_