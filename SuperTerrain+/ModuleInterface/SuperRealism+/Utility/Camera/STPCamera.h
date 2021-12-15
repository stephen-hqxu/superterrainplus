#pragma once
#ifndef _STP_CAMERA_H_
#define _STP_CAMERA_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include "../../Environment/STPCameraSetting.h"

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <utility>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPCamera is a basic camera for OpenGL.
	*/
	class STP_REALISM_API STPCamera {
	public:

		/**
		 * @brief STPMoveDirection defines in which direction the camera should move to.
		*/
		enum class STPMoveDirection : unsigned char {
			Forward = 0x00u,
			Backward = 0x01u,
			Left = 0x10u,
			Right = 0x11u,
			Up = 0x20u,
			Down = 0x21u
		};

		//A matrix and a status.
		//The true status indicates the matrix is the same compared to last time it was queried.
		typedef std::pair<const glm::mat4*, bool> STPMatrixResult;

	private:

		//The view matrix to transform from world space to view space.
		mutable glm::mat4 View;
		mutable bool ViewOutdated;

		/**
		 * @brief Update the camera vectors.
		*/
		void updateViewSpace();

	protected:

		STPEnvironment::STPCameraSetting Camera;

		//A vector defines to the up and right of the camera
		glm::vec3 Front, Up, Right;
		glm::vec2 LastRotateOffset;

	public:

		/**
		 * @brief Initialise a new camera with user-defined settings.
		 * @param props The pointer to the initial camera settings.
		 * Settings are copied to the camera class.
		*/
		STPCamera(const STPEnvironment::STPCameraSetting&);

		STPCamera(const STPCamera&) = default;

		STPCamera(STPCamera&&) noexcept = default;

		STPCamera& operator=(const STPCamera&) = default;

		STPCamera& operator=(STPCamera&&) noexcept = default;

		virtual ~STPCamera() = default;

		/**
		 * @brief Get the camera view matrix that transform from world space to view space.
		 * @return The the view matrix result.
		*/
		STPMatrixResult view() const;

		/**
		 * @brief Get the camera projection matrix that transform from view to clip space.
		 * @return The pointer to the projection matrix.
		*/
		virtual STPMatrixResult projection() const = 0;

		/**
		 * @brief Get the current camera status.
		 * @return The pointer to the current camera status.
		*/
		const STPEnvironment::STPCameraSetting& cameraStatus() const;

		/**
		 * @brief Move position of the camera in the world.
		 * @param direction Specify in which direction the camera should be moved to.
		 * @param delta A multiplier to the movement speed.
		*/
		void move(STPMoveDirection, float);

		/**
		 * @brief Rotate the orientation of the camera in the world.
		 * @param offset The relative angle of offset of the camera.
		*/
		void rotate(glm::vec2);

	};

}
#endif//_STP_CAMERA_H_