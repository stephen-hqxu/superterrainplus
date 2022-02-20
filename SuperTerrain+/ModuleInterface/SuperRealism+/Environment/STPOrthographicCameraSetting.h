#pragma once
#ifndef _STP_ORTHOGRAPHIC_CAMERA_SETTING_H_
#define _STP_ORTHOGRAPHIC_CAMERA_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include <SuperTerrain+/Environment/STPSetting.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPOrthographicCameraSetting stores settings for orthographic camera.
	*/
	struct STP_REALISM_API STPOrthographicCameraSetting : public STPSetting {
	public:

		//Define the bounding surface.
		double Left, Right, Bottom, Top;

		/**
		 * @brief Init a STPOrthographicCameraSetting with default settings.
		*/
		STPOrthographicCameraSetting();

		~STPOrthographicCameraSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_ORTHOGRAPHIC_CAMERA_SETTING_H_