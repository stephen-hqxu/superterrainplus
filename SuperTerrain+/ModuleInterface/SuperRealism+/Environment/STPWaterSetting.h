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
	struct STP_REALISM_API STPWaterSetting {
	public:

		/**
		 * @brief STPWaterWaveSetting controls the procedurally generated water wave animation.
		*/
		struct STPWaterWaveSetting {
		public:

			//Specifies the initial water wave parameters.
			float InitialRotation, InitialFrequency, InitialAmplitude, InitialSpeed;
			//Specifies how each the wave parameter evolve after each octave.
			float OctaveRotation, Lacunarity, Persistence, OctaveSpeed;
			//Specifies the wave drag. Drag controls how fast wave maxima decays as the distance to it increases.
			//Higher drag creates a more ridged wave.
			float WaveDrag;

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
		//Specifies the height multiplier for the water plane.
		//It is recommended that water altitude should be no more than the terrain altitude.
		float Altitude;

		//Controls the tessellation behaviour of the water plane mesh.
		STPTessellationSetting WaterMeshTess;
		//Controls the animation of water wave.
		STPWaterWaveSetting WaterWave;

		//The height multiplier to the water wave.
		float WaveHeight;

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
		//Controls water normalmap generation, this scales the distance of each sampling point.
		//Large distance gives smoother normalmap while small one gives stronger.
		float NormalEpsilon;

		void validate() const;

	};

}
#endif//_STP_WATER_SETTING_H_