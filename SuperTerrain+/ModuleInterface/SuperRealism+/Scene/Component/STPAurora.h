#pragma once
#ifndef _STP_AURORA_H_
#define _STP_AURORA_H_

#include <SuperRealism+/STPRealismDefine.h>
#include "../../Environment/STPAuroraSetting.h"
//Renderer
#include "../STPSceneObject.hpp"
#include "STPSkybox.h"
#include "../Light/STPLightSpectrum.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAurora procedurally generates and simulates northern light effect using procedural noise technique.
	 * It utilises triangular noise to generate a sharp-fin like pattern and distorts the noise to create aurora striping and tailing effects.
	*/
	class STP_REALISM_API STPAurora : private STPSkybox, public STPSceneObject::STPEnvironmentObject, public STPSceneObject::STPAnimatedObject {
	private:

		//aurora colour effects
		const STPLightSpectrum AuroraSpectrum;

		STPOpenGL::STPuint AuroraTimeLocation;

	public:

		/**
		 * @brief Initialise an aurora shading object.
		 * @param aurora_spectrum The colour spectrum used for rendering the aurora.
		 * The colour is picked based on the aurora's height, from 0.0 to 1.0.
		 * @param aurora_init The initialiser for the aurora environment renderer.
		*/
		STPAurora(STPLightSpectrum&&, const STPSkyboxInitialiser&);

		STPAurora(const STPAurora&) = delete;

		STPAurora(STPAurora&&) = delete;

		STPAurora& operator=(const STPAurora&) = delete;

		STPAurora& operator=(STPAurora&&) = delete;

		~STPAurora() = default;

		/**
		 * @brief Update the aurora renderer settings.
		 * @param aurora_setting The new setting.
		*/
		void setAurora(const STPEnvironment::STPAuroraSetting&);

		void updateAnimationTimer(double) override;

		void render() const override;

	};
}
#endif//_STP_AURORA_H_