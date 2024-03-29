#pragma once
#ifndef _STP_AURORA_SETTING_H_
#define _STP_AURORA_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPAuroraSetting stores settings for the procedurally generated aurora effect.
	*/
	struct STP_REALISM_API STPAuroraSetting {
	public:

		/**
		 * @brief STPTriangularNoiseSetting configures triangular noise generation.
		 * Triangular noise is the main noise function to generate a sky of auroras.
		 * Fractal triangular noise is done by composing a main noise and a distortion noise together.
		*/
		struct STP_REALISM_API STPTriangularNoiseSetting {
		public:

			/**
			 * @brief STPNoiseFractalSetting specifies parameters for the fractal triangular noise functions.
			*/
			struct STP_REALISM_API STPNoiseFractalSetting {
			public:

				//The initial amplitude 
				float InitialAmplitude;
				//The amplitude multiplier at each octave.
				float Persistence;
				//The frequency multiplier at each octave.
				float Lacunarity;

				void validate() const;

			};
			
			//Fractal settings for the main noise and the distortion noise applied on to the main noise.
			STPNoiseFractalSetting MainNoise, DistortionNoise;
			//Specifies the initial frequency for the distortion noise.
			float InitialDistortionFrequency;
			//A curvature angle in radians.
			//Instead of having a straight noise strip, the strip will be curved by the angle specified.
			float Curvature;
			//Specifies the angle of rotation in radians at every octave.
			float OctaveRotation;
			//The speed of animation.
			float AnimationSpeed;
			//The contrast of the noise value.
			//Higher value darkens areas with low noise value and amplifies bright areas.
			float Contrast;
			//Clamp the output noise intensity within the maximum range as specified.
			//The lower range is zero.
			float MaximumIntensity;
			//Specifies the number of octave.
			unsigned int Octave;

			void validate() const;

		};

		//The main aurora noise generation function.
		STPTriangularNoiseSetting Noise;
		//Specifies the flatness of the aurora sphere where aurora is generated.
		//Altitude is emulated by expanding / contracting the space in between a sphere or oval.
		//Smaller value makes the sphere flatter, turning into a pie.
		float AuroraSphereFlatness;
		//A bias value used when projecting a sphere space to a plane.
		//The bias value rotates the sphere to avoid having the plane extended to infinity at horizon, i.e., division by zero.
		//It rotates the sphere towards negative y axis.
		float AuroraPlaneProjectionBias;
		//The size at each iteration of generation.
		//Higher step size gives longer tails but may create colour banding.
		float StepSize;
		//Specifies a range when aurora starts fading out and completely invisible.
		//The altitude is specified as ray direction in range [0.0, 1.0].
		float AltitudeFadeStart, AltitudeFadeEnd;
		//The aurora intensity.
		float LuminosityMultiplier;
		//The number of iteration.
		//Greater number of iteration reduces colour banding effect but increases computational time.
		unsigned int Iteration;

		void validate() const;

	};
}
#endif//_STP_AURORA_SETTING_H_