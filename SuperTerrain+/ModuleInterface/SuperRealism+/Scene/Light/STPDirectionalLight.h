#pragma once
#ifndef _STP_DIRECTIONAL_LIGHT_H_
#define _STP_DIRECTIONAL_LIGHT_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Light
#include "../STPSceneLight.h"
//Light Setting
#include "../../Environment/STPLightSetting.h"
//Directional Light Shadow Solution
#include "STPCascadedShadowMap.h"

#include <optional>

#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief Directional light is a type of global light that emits parallel light from undefined position, but rather,
	 * it has a defined direction denoted by a unit vector.
	*/
	class STP_REALISM_API STPDirectionalLight : public STPSceneLight {
	public:

		typedef std::optional<STPCascadedShadowMap> STPDirectionalLightShadow;

	private:

		STPDirectionalLightShadow Shadow;

		//Mapped pointer
		glm::vec3* Dir;
		float* DirSpecCoord;

	public:

		/**
		 * @brief Init a STPDirectionalLight instance.
		 * @param dir_shadow The rvalue reference to the directional light shadow instance.
		 * Alternatively, a nullptr can be provided to indicate this directional light instance should not cast shadow.
		 * @see STPSceneLight
		*/
		STPDirectionalLight(STPDirectionalLightShadow&&, STPLightSpectrum&&);

		~STPDirectionalLight() = default;

		const STPLightShadow* getLightShadow() const override;

		void setSpectrumCoordinate(float) override;

		/**
		 * @brief Set the directional light property.
		 * @param directional The pointer to the directional light property.
		*/
		void setDirectional(const STPEnvironment::STPLightSetting::STPDirectionalLightSetting&);

		/**
		 * @brief Set the light direction.
		 * @param dir The light direction to be set.
		*/
		void setLightDirection(const glm::vec3&);

	};

}
#endif//_STP_DIRECTIONAL_LIGHT_H_