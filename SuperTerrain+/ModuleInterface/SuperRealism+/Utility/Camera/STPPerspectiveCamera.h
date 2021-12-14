#pragma once
#ifndef _STP_PERSPECTIVE_CAMERA_H_
#define _STP_PERSPECTIVE_CAMERA_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Camera
#include "STPCamera.h"

//GLM
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPPerspectiveCamera is a camera with perspective project.
	*/
	class STP_REALISM_API STPPerspectiveCamera : public STPCamera {
	public:

		/**
		 * @brief STPProjectionProperty stores settings for perspective projection
		*/
		struct STP_REALISM_API STPProjectionProperty {
		public:

			//Controls the FoV of the perspective camera and the zooming speed. Angle is defined in radians.
			float ViewAngle, ZoomSensitivity;
			//The min and max allowed zooming angle, in radians.
			glm::vec2 ZoomLimit;
			//Clipping planes
			float Aspect, Near, Far;

			/**
			 * @brief Initialise a perspective projection property with default settings.
			*/
			STPProjectionProperty();

			~STPProjectionProperty() = default;

		};

	private:

		STPProjectionProperty PerspectiveFrustum;
		//The projection matrix
		mutable glm::mat4 PerspectiveProjection;
		//Denotes if the projection matrix is no longer correct with the current setting.
		mutable bool ProjectionOutdated;

	public:

		/**
		 * @brief Initialise a new perspective camera with settings.
		 * @param projection_props The pointer to the settings for perspective projection.
		 * This argument is copied to the current instance.
		 * @param camera_props The settings for the base camera.
		*/
		STPPerspectiveCamera(const STPProjectionProperty&, const STPCamera::STPCameraProperty&);

		STPPerspectiveCamera(const STPPerspectiveCamera&) = default;

		STPPerspectiveCamera(STPPerspectiveCamera&&) noexcept = default;

		STPPerspectiveCamera& operator=(const STPPerspectiveCamera&) = default;

		STPPerspectiveCamera& operator=(STPPerspectiveCamera&&) noexcept = default;

		~STPPerspectiveCamera() = default;

		/**
		 * @brief Get the current perspective projection frustum status.
		 * @return The pointer to the current perspective view frustum status.
		*/
		const STPProjectionProperty& perspectiveStatus() const;

		/**
		 * @brief Get the perspective projection matrix.
		 * @return The current perspective projection matrix result.
		*/
		STPMatrixResult perspective() const;

		STPMatrixResult projection() const override;

		/**
		 * @brief Zoom the view frustum to change the perspective field-of-view.
		 * @param delta The angle to be changed by, in radians.
		*/
		void zoom(float);

		/**
		 * @brief Rescale the view frustum by changing the aspect ratio.
		 * @param aspect The new aspect ratio.
		*/
		void rescale(float);

		/**
		 * @brief Reshape the view frustum, specifically the near and far plane for the perspective view frustum.
		 * @param shape The new shape, with near and far plane distance respectively in each component.
		*/
		void reshape(glm::vec2);

	};

}
#endif//_STP_PERSPECTIVE_CAMERA_H_