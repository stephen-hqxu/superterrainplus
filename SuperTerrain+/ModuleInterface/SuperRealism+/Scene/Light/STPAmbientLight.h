#pragma once
#ifndef _STP_AMBIENT_LIGHT_H_
#define _STP_AMBIENT_LIGHT_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Light
#include "../STPSceneLight.h"
//Light Setting
#include "../../Environment/STPLightSetting.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief Ambient light is a type of global light source that emulates indirect light colour coming from all different lights.
	 *  It emits a dim constant light from unspecified location, thus it never leaves shadows, but instead it contributes to
	 * ambient occlusion.
	*/
	class STP_REALISM_API STPAmbientLight : public STPSceneLight {
	private:

		//A mapped pointer to ambient light spectrum sampling coordinate.
		float* AmbSpecCoord;

	public:

		/**
		 * @brief Initialise a new ambient light instance.
		 * @see STPSceneLight
		*/
		STPAmbientLight(STPLightSpectrum&&);

		~STPAmbientLight() = default;

		const STPLightShadow* getLightShadow() const override;

		void setSpectrumCoordinate(float) override;

		/**
		 * @brief Set the ambient light property.
		 * @param ambient The pointer to the ambient light property.
		*/
		void setAmbient(const STPEnvironment::STPLightSetting::STPAmbientLightSetting&);

	};

}
#endif//_STP_AMBIENT_LIGHT_H_