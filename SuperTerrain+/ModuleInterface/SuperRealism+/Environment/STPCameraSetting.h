#pragma once
#ifndef _STP_CAMERA_SETTING_H_
#define _STP_CAMERA_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPCameraSetting stores parameters for the camera.
	*/
	struct STP_REALISM_API STPCameraSetting {
	public:

		//Euler's angles to define the rotation of the camera, expressed in radians.
		double Yaw, Pitch;
		//Controls the view angle of the camera, in radians.
		double FoV;
		//These parameters define the animation of the camera.
		double MovementSpeed, RotationSensitivity, ZoomSensitivity;

		//The min and max allowed zooming angle, in radians.
		glm::dvec2 ZoomLimit;
		//These vectors define a camera coordinate in world space and can be used to construct view matrix.
		glm::dvec3 Position, WorldUp;

		//Frustum aspect ratio.
		double Aspect;
		//Define the depth of the view frustum.
		double Near, Far;

		void validate() const;

	};

}
#endif//_STP_CAMERA_SETTING_H_