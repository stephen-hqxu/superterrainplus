#pragma once
#ifndef _STP_SCENE_LIGHT_H_
#define _STP_SCENE_LIGHT_H_

#include <SuperRealism+/STPRealismDefine.h>
//Light
#include "./Light/STPLightSpectrum.h"
//Light Shadow
#include "./Light/STPLightShadow.h"
//GL Object
#include "../Object/STPBindlessBuffer.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSceneLight is a type of objects that emits light, it is a collection of all different types of light source.
	 * Most light sources in the scene are allowed to be chosen between a shadow-casting and non shadow-casting light.
	*/
	class STP_REALISM_API STPSceneLight {
	public:

		/**
		 * @brief STPLightType identifies the type of a light.
		*/
		enum class STPLightType : unsigned char {
			Ambient = 0x00u,
			Directional = 0x01u
		};

	protected:

		//Please do note that all optional fields are required to be initialised before rendering.

		//A buffer stores data of a light.
		STPBuffer LightData;
		//An address to the light data buffer to be shared with shaders.
		std::optional<STPBindlessBuffer> LightDataAddress;

	public:

		const STPLightType Type;
		//The light spectrum specifies the colour of the light
		const STPLightSpectrum LightSpectrum;

		/**
		 * @brief Init a STPSceneLight instance.
		 * @param spectrum The light spectrum used for this light instance.
		 * This light spectrum will be moved under the current light instance upon construction.
		 * @param type Specifies the type of light.
		*/
		STPSceneLight(STPLightSpectrum&&, STPLightType);

		virtual ~STPSceneLight();

		/**
		 * @brief Get the pointer to the light shadow instance.
		 * @return The pointer to the light shadow of the current light.
		 * The pointer might be null to denote that this light should not cast any shadow.
		*/
		virtual const STPLightShadow* getLightShadow() const = 0;

		/**
		 * @brief Set the sampling coordinate of the light spectrum texture.
		 * @param coord The spectrum coordinate.
		*/
		virtual void setSpectrumCoordinate(float) = 0;

		/**
		 * @see getLightShadow() const
		*/
		STPLightShadow* getLightShadow();

		/**
		 * @brief Get the address to the light data buffer.
		 * @return The light data buffer address.
		*/
		STPOpenGL::STPuint64 lightDataAddress() const;

	};

}
#endif//_STP_SCENE_LIGHT_H_