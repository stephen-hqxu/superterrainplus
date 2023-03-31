#pragma once
#ifndef _STP_STARFIELD_H_
#define _STP_STARFIELD_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include "../../Environment/STPStarfieldSetting.h"
//Environment Renderer
#include "../STPSceneObject.hpp"
#include "STPSkybox.h"
#include "../Light/STPLightSpectrum.h"

#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPStarfield simulates a procedural starfield for night rendering.
	 * It utilises simple grid division algorithm and assign a random likelihood value for each grid,
	 * and generates stars with soft falloff from the centre of the grid.
	*/
	class STP_REALISM_API STPStarfield : public STPSceneObject::STPEnvironmentObject, public STPSceneObject::STPAnimatedObject {
	public:

		/**
		 * @brief STPStarfieldModel specifies the behaviour of the procedural model for the starfield generator.
		*/
		struct STPStarfieldModel {
		public:

			//Specifies a colour spectrum for starlight.
			//The light spectrum will be moved upon initialisation.
			STPLightSpectrum* StarlightSpectrum;
			//Specifies if the star intensity attenuation should be quadratic.
			//This will suppress very dark stars and help decreasing the star density.
			bool UseQuadraticAttenuation;

		};

	private:

		STPSkybox StarfieldBox;

		//and a colourful starfield
		const STPLightSpectrum StarlightSpectrum;

		//Recorded uniform locations
		STPOpenGL::STPint ShineTimeLocation;

	public:

		/**
		 * @brief Initialise a procedural starfield renderer.
		 * @param starfield_model Specifies the starfield generation model to control the behaviour.
		 * @param starfield_init The initialiser for the starfield renderer.
		*/
		STPStarfield(const STPStarfieldModel&, const STPSkybox::STPSkyboxInitialiser&);

		STPStarfield(const STPStarfield&) = delete;

		STPStarfield(STPStarfield&&) = delete;

		STPStarfield& operator=(const STPStarfield&) = delete;

		STPStarfield& operator=(STPStarfield&&) = delete;

		~STPStarfield() = default;

		/**
		 * @brief Update the starfield renderer settings.
		 * @param starfield_setting The new setting.
		*/
		void setStarfield(const STPEnvironment::STPStarfieldSetting&);

		void updateAnimationTimer(double) override;

		void render() const override;

	};
}
#endif//_STP_STARFIELD_H_