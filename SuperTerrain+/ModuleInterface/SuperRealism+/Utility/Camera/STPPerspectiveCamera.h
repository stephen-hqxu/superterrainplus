#pragma once
#ifndef _STP_PERSPECTIVE_CAMERA_H_
#define _STP_PERSPECTIVE_CAMERA_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Camera
#include "STPCamera.h"
#include "../../Environment/STPPerspectiveCameraSetting.h"

//GLM
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPPerspectiveCamera is a camera with perspective projection.
	*/
	class STP_REALISM_API STPPerspectiveCamera : public STPCamera {
	private:

		//The projection matrix
		alignas(32) mutable glm::dmat4 PerspectiveProjection;
		STPEnvironment::STPPerspectiveCameraSetting Frustum;
		//Denotes if the projection matrix is no longer correct with the current setting.
		mutable bool ProjectionOutdated;

		/**
		 * @brief Trigger outdated event.
		*/
		void setOutdated();

	public:

		/**
		 * @brief Initialise a new perspective camera with settings.
		 * @param projection_props The pointer to the settings for perspective projection.
		 * This argument is copied to the current instance.
		 * @param camera_props The settings for the base camera.
		*/
		STPPerspectiveCamera(const STPEnvironment::STPPerspectiveCameraSetting&, const STPEnvironment::STPCameraSetting&);

		STPPerspectiveCamera(const STPPerspectiveCamera&) = default;

		STPPerspectiveCamera(STPPerspectiveCamera&&) noexcept = default;

		STPPerspectiveCamera& operator=(const STPPerspectiveCamera&) = default;

		STPPerspectiveCamera& operator=(STPPerspectiveCamera&&) noexcept = default;

		~STPPerspectiveCamera() = default;

		/**
		 * @brief Get the current perspective projection frustum status.
		 * @return The pointer to the current perspective view frustum status.
		*/
		const STPEnvironment::STPPerspectiveCameraSetting& perspectiveStatus() const;

		/**
		 * @brief Get the perspective projection matrix.
		 * @return The pointer to the current perspective matrix which is cached by the current camera.
		 * Like the base camera class, this pointer will not be updated by the instance automatically.
		*/
		const glm::dmat4& perspective() const;

		const glm::dmat4& projection() const override;

		glm::dmat4 projection(double, double) const override;

		/**
		 * @brief Zoom the view frustum to change the perspective field-of-view.
		 * @param delta The angle to be changed by, in radians.
		*/
		void zoom(double);

		/**
		 * @brief Rescale the view frustum by changing the aspect ratio.
		 * @param aspect The new aspect ratio.
		*/
		void rescale(double);

		/**
		 * @brief Reshape the view frustum, specifically the near and far plane for the perspective view frustum.
		 * @param shape The new shape, with near and far plane distance respectively in each component.
		*/
		void reshape(glm::dvec2);

	};

}
#endif//_STP_PERSPECTIVE_CAMERA_H_