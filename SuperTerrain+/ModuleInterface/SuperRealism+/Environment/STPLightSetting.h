#pragma once
#ifndef _STP_LIGHT_SETTING_H_
#define _STP_LIGHT_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Setting
#include <SuperTerrain+/Environment/STPSetting.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPLightSetting contains a collection of settings for different types of light.
	*/
	namespace STPLightSetting {

		/**
		 * @brief STPAmbientLightSetting contains settings for ambient light.
		*/
		struct STP_REALISM_API STPAmbientLightSetting : public STPSetting {
		public:

			float AmbientStrength;

			/**
			 * @brief Init a new STPAmbientLightSetting with default settings loaded.
			*/
			STPAmbientLightSetting();

			~STPAmbientLightSetting() = default;

			bool validate() const override;

		};

		/**
		 * @brief STPDirectionalLightSetting contains settings for directional light.
		*/
		struct STP_REALISM_API STPDirectionalLightSetting : public STPSetting {
		public:

			float DiffuseStrength, SpecularStrength;

			/**
			 * @brief Init a new STPDirectionalLightSetting with default settings.
			*/
			STPDirectionalLightSetting();

			~STPDirectionalLightSetting() = default;

			bool validate() const override;

		};

	}

}
#endif//_STP_LIGHT_SETTING_H_