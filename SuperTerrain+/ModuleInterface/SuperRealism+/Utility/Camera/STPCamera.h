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

	private:

		//The view matrix to transform from world space to view space.
		mutable glm::mat4 View;
		//Flag to indicate if the camera has changed its state since last time the view matrix was computed.
		//The flag will be reset until view matrix is recomputed.
		mutable bool Moved, Rotated;

		/**
		 * @brief Update the camera vectors.
		*/
		void updateViewSpace();

	protected:

		STPEnvironment::STPCameraSetting Camera;

		//A vector defines to the up and right of the camera
		glm::vec3 Front, Up, Right;

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
		 * @brief Update and get the camera view matrix that transform from world space to view space.
		 * @return The pointer to the view matrix cached by the camera.
		 * Note that this pointer will not be updated by the camera automatically unless this function is called.
		*/
		const glm::mat4& view() const;

		/**
		 * @brief Get the camera projection matrix that transform from view to clip space.
		 * @return The pointer to the projection matrix.
		*/
		virtual const glm::mat4& projection() const = 0;

		/**
		 * @brief Get the camera projection matrix that overrides the camera near and far plane settings.
		 * @param near The near plane of the projection frustum.
		 * @param far The far plane of the projection frustum.
		 * @return The projection matrix.
		*/
		virtual glm::mat4 projection(float, float) const = 0;

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
		 * @brief Check if the camera has moved since last time view matrix was updated.
		 * @return True if camera has moved.
		*/
		bool hasMoved() const;

		/**
		 * @brief Rotate the orientation of the camera in the world.
		 * @param offset The relative angle of offset of the camera.
		*/
		void rotate(const glm::vec2&);

		/**
		 * @brief Check if the camera has rotated since last time view matrix was updated.
		 * @return True if camera has rotated.
		*/
		bool hasRotated() const;

		/**
		 * @brief Check if the camera projection has its shape changed since last time projection matrix was computed.
		 * @return True if the camera shape is outdated and requires update.
		*/
		virtual bool reshaped() const = 0;

	};

}
#endif//_STP_CAMERA_H_