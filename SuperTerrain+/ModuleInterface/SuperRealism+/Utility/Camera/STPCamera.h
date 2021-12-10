#pragma once
#ifndef _STP_CAMERA_H_
#define _STP_CAMERA_H_

#include <SuperRealism+/STPRealismDefine.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPCamera is a basic camera for OpenGL.
	*/
	class STP_REALISM_API STPCamera {
	public:

		/**
		 * @brief STPCameraProperty stores current parameters for the camera.
		*/
		struct STP_REALISM_API STPCameraProperty {
		public:

			//Euler's angles to define the rotation of the camera, expressed in radians
			float Yaw, Pitch;
			//These parameters define the animation of the camera.
			float MovementSpeed, RotationSensitivity;

			//These vectors define a camera coordinate in world space and can be used to construct view matrix.
			glm::vec3 Position, WorldUp;

			/**
			 * @brief Initialise a camera property with default settings.
			*/
			STPCameraProperty();

			~STPCameraProperty() = default;

		};

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

	private:

		//The view matrix to transform from world space to view space.
		mutable glm::mat4 View;
		mutable bool ViewOutdated;

		/**
		 * @brief Update the camera vectors.
		*/
		void updateViewSpace();

	protected:

		//Properties of the camera
		STPCameraProperty Camera;

		//A vector defines to the up and right of the camera
		glm::vec3 Front, Up, Right;
		glm::vec2 LastRotateOffset;

	public:

		/**
		 * @brief Initialise a new camera with user-defined settings.
		 * @param props The pointer to the initial camera settings.
		 * Settings are copied to the camera class.
		*/
		STPCamera(const STPCameraProperty&);

		STPCamera(const STPCamera&) = default;

		STPCamera(STPCamera&&) noexcept = default;

		STPCamera& operator=(const STPCamera&) = default;

		STPCamera& operator=(STPCamera&&) noexcept = default;

		~STPCamera() = default;

		/**
		 * @brief Get the camera view matrix that transform from world space to view spalce.
		 * @return The pointer to the view matrix.
		*/
		const glm::mat4& view() const;

		/**
		 * @brief Get the current camera status.
		 * @return The pointer to the current camera status.
		*/
		const STPCameraProperty& cameraStatus() const;

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