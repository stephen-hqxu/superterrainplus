#pragma once
#ifndef _STP_CAMERA_H_
#define _STP_CAMERA_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include "../Environment/STPCameraSetting.h"
//Algebra
#include <SuperTerrain+/Utility/Algebra/STPMatrix4x4d.h>
//Camera Buffer
#include "../Object/STPBuffer.h"

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

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
		 * @brief STPSubscriberBenefit contains some data passed to the application when certain camera event happens.
		 * User can check for those flags to determine what has been changed.
		 * The flag will only be set to true when an event happens, it is the application's responsibility to unset the flags.
		*/
		struct STPSubscriberBenefit {
		public:

			//The camera has moved
			bool Moved;
			//The camera has rotated
			bool Rotated;
			//The camera has zoomed
			bool Zoomed;
			//The camera has its aspect ratio changed
			bool AspectChanged;

			/**
			 * @brief True if any of the status flag is true.
			 * @return True if any of the status flag is true.
			*/
			bool any() const noexcept;

			/**
			 * @brief Set all status flags to false.
			*/
			void unset() noexcept;

		};

	private:

		STPEnvironment::STPCameraSetting Camera;
		//A vector defines to the up and right of the camera
		glm::dvec3 Front, Up, Right;

		//The view matrix to transform from world space to view space. And the perspective projection matrix.
		STPMatrix4x4d View, PerspectiveProjection;

		//Flag to indicate if the camera has changed its state since last time the view matrix was computed.
		bool PositionOutdated, ViewOutdated, ProjectionOutdated;

		/**
		 * @brief Packed structure for mapped camera buffer following OpenGL std430 alignment rule.
		*/
		struct STPPackedCameraBuffer;
		//send all camera data to GPU so they are be accessed from shaders
		STPBuffer CameraInformation;
		STPPackedCameraBuffer* MappedBuffer;

		std::vector<STPSubscriberBenefit*> CameraSubscriber;

		/**
		 * @brief Update the camera vectors.
		*/
		void updateViewSpace();

		/**
		 * @brief Trigger a camera event for each member in the subscriber.
		 * @tparam Func The type of event function.
		 * @param func The event function.
		*/
		template<class Func>
		void triggerSubscriberEvent(Func&&) const;

		/**
		 * @brief Find a subscriber instance in the database.
		 * @param benefit The pointer to subscriber to be found.
		 * @return The iterator in the registry.
		*/
		auto findSubcriber(STPSubscriberBenefit*) const;

	public:

		/**
		 * @brief Initialise a new camera with user-defined settings.
		 * @param props The pointer to the initial camera settings.
		 * Settings are copied to the camera class.
		*/
		STPCamera(const STPEnvironment::STPCameraSetting&);

		STPCamera(const STPCamera&) = delete;

		STPCamera(STPCamera&&) noexcept = default;

		STPCamera& operator=(const STPCamera&) = delete;

		STPCamera& operator=(STPCamera&&) noexcept = default;

		~STPCamera() = default;

		/**
		 * @brief Subscribe to a camera status change.
		 * @param benefit A packet of data from user to receive event status.
		 * This packet must remain valid until it is unsubscribed.
		*/
		void subscribe(STPSubscriberBenefit&);

		/**
		 * @brief Remove a previously subscribed listener from the camera.
		 * @param benefit The packet instance to be removed.
		*/
		void unsubscribe(STPSubscriberBenefit&);

		/**
		 * @brief Bind the camera buffer to a target GL buffer.
		 * @param target The target to bind to.
		 * @param index The index of the target buffer.
		*/
		void bindCameraBuffer(STPOpenGL::STPenum, STPOpenGL::STPuint) const;

		/**
		 * @brief Flush all updated states to the memory.
		 * This operation is expensive, so it is the best to batch all updates together and call this function once per rendering frame.
		 * Not calling this function before rendering will cause potential race condition so UB.
		*/
		void flush();

		/**
		 * @brief Get the last flushed camera view matrix that transform from world space to view space.
		 * @return The pointer to the view matrix cached by the camera.
		 * Note that this pointer will not be updated by the camera automatically unless the flush function is called.
		 * The pointer of the returned view matrix is aligned to 32-bit boundary.
		*/
		const STPMatrix4x4d& view() const noexcept;

		/**
		 * @brief Get the last flushed camera projection matrix that transform from view to clip space.
		 * @return The pointer to the projection matrix.
		 * Note that this pointer will not be updated by the camera automatically unless the flush function is called.
		 * The pointer of the returned projection matrix is aligned to 32-bit boundary.
		*/
		const STPMatrix4x4d& projection() const noexcept;

		/**
		 * @brief Get the camera projection matrix that overrides the camera near and far plane settings.
		 * @param near The near plane of the projection frustum.
		 * @param far The far plane of the projection frustum.
		 * @return The projection matrix.
		*/
		STPMatrix4x4d projection(double, double) const;

		/**
		 * @brief Get the current camera status.
		 * @return The pointer to the current camera status.
		*/
		const STPEnvironment::STPCameraSetting& cameraStatus() const noexcept;

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

		/**
		 * @brief Zoom the view frustum to change the perspective field-of-view.
		 * @param delta The angle to be changed by, in radians.
		*/
		void zoom(double);

		/**
		 * @brief Change the aspect ratio.
		 * @param aspect The new aspect ratio.
		*/
		void setAspect(double);

	};

}
#endif//_STP_CAMERA_H_