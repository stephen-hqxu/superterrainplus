#pragma once
#ifndef _STP_PERSPECTIVE_CAMERA_SETTING_H_
#define _STP_PERSPECTIVE_CAMERA_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base
#include <SuperTerrain+/Environment/STPSetting.hpp>

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPPerspectiveCameraSetting stores settings for perspective camera projection
	*/
	struct STP_REALISM_API STPPerspectiveCameraSetting : public STPSetting {
	public:

		//Controls the FoV of the perspective camera and the zooming speed. Angle is defined in radians.
		double ViewAngle, ZoomSensitivity;
		//The min and max allowed zooming angle, in radians.
		glm::dvec2 ZoomLimit;
		//Frustum aspect ratio
		double Aspect;

		/**
		 * @brief Init STPPerspectiveCameraSetting with default settings.
		*/
		STPPerspectiveCameraSetting();

		~STPPerspectiveCameraSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_PERSPECTIVE_CAMERA_SETTING_H_