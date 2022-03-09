#pragma once
#ifndef _STP_CAMERA_SETTING_H_
#define _STP_CAMERA_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Setting
#include <SuperTerrain+/Environment/STPSetting.hpp>

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPCameraSetting stores parameters for the camera.
	*/
	struct STP_REALISM_API STPCameraSetting : public STPSetting {
	public:

		//Euler's angles to define the rotation of the camera, expressed in radians
		double Yaw, Pitch;
		//These parameters define the animation of the camera.
		double MovementSpeed, RotationSensitivity;

		//These vectors define a camera coordinate in world space and can be used to construct view matrix.
		glm::dvec3 Position, WorldUp;

		//Define the depth of the view frustum.
		double Near, Far;

		/**
		 * @brief Init STPCameraSetting with default values.
		*/
		STPCameraSetting();

		~STPCameraSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_CAMERA_SETTING_H_