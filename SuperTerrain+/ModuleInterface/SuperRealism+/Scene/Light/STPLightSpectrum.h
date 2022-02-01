#pragma once
#ifndef _STP_LIGHT_SPECTRUM_H_
#define _STP_LIGHT_SPECTRUM_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../../Object/STPTexture.h"

//GLM
#include <glm/vec3.hpp>

//System
#include <vector>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLightSpectrum allows generating a light spectrum for looking up color for indirect and direct lighting.
	*/
	class STP_REALISM_API STPLightSpectrum {
	public:

		/**
		 * @brief Specify the type of spectrum.
		*/
		enum class STPSpectrumType : unsigned char {
			//The spectrum only has one color strip.
			//The spectrum will have the type of a 1D texture.
			Monotonic = 0x00u,
			//The spectrum has two color strips.
			//The spectrum will have the type of a 1D texture array.
			Bitonic = 0x01u
		};

	protected:

		//The generated spectrum, it is a texture 1D array with the first array being the spectrum for indirect lighting and the second for direct lighting
		STPTexture Spectrum;

	public:

		//Record the length of the spectrum, i.e., the number of pixel each array element has.
		const unsigned int SpectrumLength;

		/**
		 * @brief Init a light spectrum object.
		 * @param length The length of the light spectrum.
		 * @param type The type of the spectrum.
		 * @param format Specify the sized channel format for the spectrum.
		*/
		STPLightSpectrum(unsigned int, STPSpectrumType, STPOpenGL::STPenum);

		STPLightSpectrum(const STPLightSpectrum&) = delete;

		STPLightSpectrum(STPLightSpectrum&&) noexcept = default;

		STPLightSpectrum& operator=(const STPLightSpectrum&) = delete;

		STPLightSpectrum& operator=(STPLightSpectrum&&) noexcept = default;

		virtual ~STPLightSpectrum() = default;

		/**
		 * @brief Get the light spectrum.
		 * @return The pointer to a GL 1D array texture.
		 * The first array contains the spectrum for indirect lighting while the second array contains that for direct lighting.
		*/
		const STPTexture& spectrum() const;

		/**
		 * @brief Get the spectrum sampling coordinate.
		 * @return The sample texture coordinate for the spectrum.
		*/
		virtual float coordinate() const = 0;

	};

	/**
	 * @brief STPStaticLightSpectrum is a simple implementation of light spectrum.
	 * It provides a single monotonic color for lighting.
	*/
	class STP_REALISM_API STPStaticLightSpectrum : public STPLightSpectrum {
	public:

		/**
		 * @brief Init a new static light spectrum with no color.
		*/
		STPStaticLightSpectrum();

		STPStaticLightSpectrum(const STPStaticLightSpectrum&) = delete;

		STPStaticLightSpectrum(STPStaticLightSpectrum&&) noexcept = default;

		STPStaticLightSpectrum& operator=(const STPStaticLightSpectrum&) = delete;

		STPStaticLightSpectrum& operator=(STPStaticLightSpectrum&&) noexcept = default;

		~STPStaticLightSpectrum() = default;

		/**
		 * @brief Set the color of the static light spectrum.
		 * @param color The color for the spectrum.
		*/
		void operator()(glm::vec3);

		float coordinate() const override;

	};

	/**
	 * @brief STPArrayLightSpectrum allows specifying light spectrum with custom color array with one color channel.
	*/
	class STP_REALISM_API STPArrayLightSpectrum : public STPLightSpectrum {
	private:

		float SampleCoordinate;

	public:

		//Contains an array of color
		typedef std::vector<glm::vec3> STPColorArray;

		/**
		 * @brief Init a new array light spectrum.
		 * @param length The length of the light spectrum.
		*/
		STPArrayLightSpectrum(unsigned int);

		STPArrayLightSpectrum(const STPArrayLightSpectrum&) = delete;

		STPArrayLightSpectrum(STPArrayLightSpectrum&&) noexcept = default;

		STPArrayLightSpectrum& operator=(const STPArrayLightSpectrum&) = delete;

		STPArrayLightSpectrum& operator=(STPArrayLightSpectrum&&) noexcept = default;

		~STPArrayLightSpectrum() = default;

		/**
		 * @brief Set the light spectrum with new array of colors.
		 * @param color The array of color to be filled into the spectrum.
		*/
		void operator()(const STPColorArray&);

		/**
		 * @brief Set the sampling coordinate on the light spectrum.
		 * @param coord The sampling coordinate.
		*/
		void setCoordinate(float);

		float coordinate() const override;

	};

}
#endif//_STP_LIGHT_SPECTRUM_H_