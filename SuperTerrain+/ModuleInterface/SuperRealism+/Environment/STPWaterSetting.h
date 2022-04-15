#pragma once
#ifndef _STP_WATER_SETTING_H_
#define _STP_WATER_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

#include "STPTessellationSetting.h"

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPWaterSetting contains settings for rendering photorealistic water.
	*/
	struct STP_REALISM_API STPWaterSetting : public STPSetting {
	public:

		/**
		 * @brief STPWaterWaveSetting controls the procedurally generated water wave animation.
		*/
		struct STP_REALISM_API STPWaterWaveSetting : public STPSetting {
		public:

			//Specifies the initial water wave parameters.
			float InitialRotation, InitialFrequency, InitialAmplitude, InitialSpeed;
			//Specifies how each the wave parameter evolve after each octave.
			float OctaveRotation, Lacunarity, Persistence, OctaveSpeed;
			//Specifies the wave drag. Drag controls how fast wave maxima decays as the distance to it increases.
			//Higher drag creates a more ridged wave.
			float WaveDrag;

			/**
			 * @brief Initialise a new STPWaterWaveSetting.
			*/
			STPWaterWaveSetting();

			~STPWaterWaveSetting() = default;

			bool validate() const override;

		};

		//The minimum height water plane should be displayed.
		//If the water level goes below this boundary, water plane geometry will be culled.
		float MinimumWaterLevel;
		//Specifies the number of sampling point taken when culling water geometry.
		//Higher sample count gives better result and can potentially avoid false positive which causes false-culling.
		//Lower sample count is fast, obviously.
		unsigned int CullTestSample;
		//Specifies the sample distances from the testing geometry.
		float CullTestRadius;

		//Controls the tessellation behaviour of the water plane mesh.
		STPTessellationSetting WaterMeshTess;
		//Controls the animation of water wave.
		STPWaterWaveSetting WaterWave;

		//Controls the number of iteration when procedurally generating water wave.
		//It is preferred to have fewer iterations for geometry pass and more for normal pass.
		struct {
		public:

			//Controls the number of iteration when generating the water plane geometry.
			unsigned int Geometry;
			//Controls the number iteration for normalmap generation.
			unsigned int Normal;

		} WaterWaveIteration;

		//Set a small amount of constant colour added to the water mesh.
		//This constant colour approximates how water behaves in real life where it reflects certain wavelength of light more than the others.
		glm::vec3 Tint;

		/**
		 * @brief Initialise a new STPWaterSetting instance.
		*/
		STPWaterSetting();

		~STPWaterSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_WATER_SETTING_H_