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

//System
#include <vector>

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

		/**
		 * @brief STPProjectionCategory specifies the type of projection of the camera implementation.
		*/
		enum class STPProjectionCategory : unsigned char {
			//Perspective projection
			Perspective = 0x00u,
			//Orthographic projection
			Orthographic = 0xFFu
		};

		/**
		 * @brief STPStatusChangeCallback allows actions to be taken when the camera status has changed outside of the environment.
		*/
		class STPStatusChangeCallback {
		public:

			/**
			 * @brief Default init STPStatusChangeCallback.
			*/
			STPStatusChangeCallback() = default;

			virtual ~STPStatusChangeCallback() = default;

			/**
			 * @brief The camera has moved.
			 * @param camera The pointer to the camera where it is called.
			*/
			virtual void onMove(const STPCamera&) = 0;

			/**
			 * @brief The camera has moved.
			 * @param camera The pointer to the camera where it is called.
			*/
			virtual void onRotate(const STPCamera&) = 0;

			/**
			 * @brief The camera has reshaped.
			 * @param camera The pointer to the camera where it is called.
			*/
			virtual void onReshape(const STPCamera&) = 0;

		};

	private:

		//The view matrix to transform from world space to view space.
		mutable glm::dmat4 View;
		//Flag to indicate if the camera has changed its state since last time the view matrix was computed.
		//The flag will be reset until view matrix is recomputed.
		mutable bool ViewOutdated;

		/**
		 * @brief Update the camera vectors.
		*/
		void updateViewSpace();

		/**
		 * @brief Find a listener instance in the callback registry.
		 * @param listener The pointer to listener to be found.
		 * @return The iterator in the registry.
		*/
		auto findListener(STPStatusChangeCallback*) const;

	protected:

		STPEnvironment::STPCameraSetting Camera;

		//A vector defines to the up and right of the camera
		glm::dvec3 Front, Up, Right;

		mutable std::vector<STPStatusChangeCallback*> CallbackRegistry;

	public:

		//Specifies the type of projection this camera uses.
		const STPProjectionCategory ProjectionType;

		/**
		 * @brief Initialise a new camera with user-defined settings.
		 * @param props The pointer to the initial camera settings.
		 * Settings are copied to the camera class.
		 * @param proj_type The type of projection the camera class uses.
		*/
		STPCamera(const STPEnvironment::STPCameraSetting&, STPProjectionCategory);

		STPCamera(const STPCamera&) = default;

		STPCamera(STPCamera&&) noexcept = default;

		STPCamera& operator=(const STPCamera&) = default;

		STPCamera& operator=(STPCamera&&) noexcept = default;

		virtual ~STPCamera() = default;

		/**
		 * @brief Register a camera status change listener.
		 * @param listener The listener instance to receive update.
		 * It is not allowed to registered the same listener twice, in case that happens exception is thrown.
		*/
		void registerListener(STPStatusChangeCallback*) const;

		/**
		 * @brief Remove a previously registered listener from the camera class.
		 * @param listener The listener instance to be removed.
		 * If this listener is not previously registered, exception is thrown.
		*/
		void removeListener(STPStatusChangeCallback*) const;

		/**
		 * @brief Update and get the camera view matrix that transform from world space to view space.
		 * @return The pointer to the view matrix cached by the camera.
		 * Note that this pointer will not be updated by the camera automatically unless this function is called.
		*/
		const glm::dmat4& view() const;

		/**
		 * @brief Get the camera projection matrix that transform from view to clip space.
		 * @return The pointer to the projection matrix.
		*/
		virtual const glm::dmat4& projection() const = 0;

		/**
		 * @brief Get the camera projection matrix that overrides the camera near and far plane settings.
		 * @param near The near plane of the projection frustum.
		 * @param far The far plane of the projection frustum.
		 * @return The projection matrix.
		*/
		virtual glm::dmat4 projection(double, double) const = 0;

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
		void move(STPMoveDirection, double);

		/**
		 * @brief Rotate the orientation of the camera in the world.
		 * @param offset The relative angle of offset of the camera.
		*/
		void rotate(const glm::dvec2&);

	};

}
#endif//_STP_CAMERA_H_