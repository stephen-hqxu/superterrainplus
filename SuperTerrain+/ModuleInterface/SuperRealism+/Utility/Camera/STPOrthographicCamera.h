#pragma once
#ifndef _STP_ORTHOGRAPHIC_CAMERA_H_
#define _STP_ORTHOGRAPHIC_CAMERA_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Camera
#include "STPCamera.h"
#include "../../Environment/STPOrthographicCameraSetting.h"

//GLM
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPOrthographicCamera is a camera with orthographic projection.
	*/
	class STP_REALISM_API STPOrthographicCamera : public STPCamera {
	private:

		//The projection matrix
		mutable STPMatrix4x4d OrthographicProjection;
		STPEnvironment::STPOrthographicCameraSetting Frustum;
		//A flag
		mutable bool ProjectionOutdated;

	public:

		/**
		 * @brief Initialise a new orthographic camera with settings.
		 * @param projection_props The pointer to the settings for orthographic projection.
		 * This argument is copied to the current instance.
		 * @param camera_props The settings for the base camera.
		*/
		STPOrthographicCamera(const STPEnvironment::STPOrthographicCameraSetting&, const STPEnvironment::STPCameraSetting&);

		STPOrthographicCamera(const STPOrthographicCamera&) = default;

		STPOrthographicCamera(STPOrthographicCamera&&) noexcept = default;

		STPOrthographicCamera& operator=(const STPOrthographicCamera&) = default;

		STPOrthographicCamera& operator=(STPOrthographicCamera&&) noexcept = default;

		~STPOrthographicCamera() = default;

		/**
		 * @brief Get the orthographic projection matrix.
		 * @return The pointer to the current orthographic matrix which is cached by the current camera.
		 * Like the base camera class, this pointer will not be updated by the instance automatically.
		*/
		const STPMatrix4x4d& ortho() const;

		const STPMatrix4x4d& projection() const override;

		STPMatrix4x4d projection(double, double) const override;

		/**
		 * @brief Reshape the view frustum. Change the bouding box vertices.
		 * @param side Specify the left, right, bottom and top of the viewer-facing side of the bounding box.
		 * @param depth Specify the near and far of the bounding box.
		*/
		void reshape(glm::dvec4, glm::dvec2);

	};

}
#endif//_STP_ORTHOGRAPHIC_CAMERA_H_