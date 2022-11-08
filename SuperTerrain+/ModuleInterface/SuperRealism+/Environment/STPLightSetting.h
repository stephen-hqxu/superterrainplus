#pragma once
#ifndef _STP_LIGHT_SETTING_H_
#define _STP_LIGHT_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPLightSetting contains a collection of settings for different types of light.
	*/
	namespace STPLightSetting {

		/**
		 * @brief STPAmbientLightSetting contains settings for ambient light.
		*/
		struct STP_REALISM_API STPAmbientLightSetting {
		public:

			float AmbientStrength;

			void validate() const;

		};

		/**
		 * @brief STPDirectionalLightSetting contains settings for directional light.
		*/
		struct STP_REALISM_API STPDirectionalLightSetting {
		public:

			float DiffuseStrength, SpecularStrength;

			void validate() const;

		};

	}

}
#endif//_STP_LIGHT_SETTING_H_