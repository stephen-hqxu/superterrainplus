#pragma once
#ifndef _STP_SCENE_LIGHT_H_
#define _STP_SCENE_LIGHT_H_

#include <SuperRealism+/STPRealismDefine.h>
//Light
#include "./Light/STPLightSpectrum.h"

//Light Shadow Solution
#include "./Light/STPCascadedShadowMap.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSceneLight is a type of objects that emits light, it is a collection of all different types of light source.
	 * Most light sources in the scene are given two options, being non-shadow-casting one (the false option) and the shadow-casting one (the true option).
	*/
	namespace STPSceneLight {

		/**
		 * @brief STPLocalLight is one type of light that has limited range of lighting effect and has clearly defined position.
		*/
		template<bool SM>
		class STPLocalLight;

		/**
		 * @brief STPGlobalLight is one type of light that emits light without attenuation to the whole scene, 
		 * and usually it does not have defined position.
		*/
		template<bool SM>
		class STPGlobalLight;

		template<>
		class STPGlobalLight<false> {
		public:

			STPGlobalLight() = default;

			virtual ~STPGlobalLight() = default;

			/**
			 * @brief Get the pointer to the light spectrum.
			 * @return The pointer to the spectrum of the current light.
			*/
			virtual const STPLightSpectrum& getLightSpectrum() const = 0;

		};

		template<>
		class STPGlobalLight<true> {
		public:

			STPGlobalLight() = default;

			virtual ~STPGlobalLight() = default;

			/**
			 * @brief Get the pointer to the light shadow instance.
			 * @return The pointer to the light shadow of the current light.
			*/
			virtual const STPLightShadow& getLightShadow() const = 0;

		};

		/**
		 * @brief Ambient light is a type of global light source that emulates indirect light color coming from all different lights.
		 * It emits a dim constant light from unspecified location, thus it never leaves shadows, but instead it contributes to
		 * ambient occlusion.
		*/
		class STPAmbientLight;

		/**
		 * @brief Directional light is a type of global light that emits parallel light from undefined position, but rather,
		 * it has a defined direction denoted by a unit vector.
		*/
		template<bool SM>
		class STPDirectionalLight;

		/**
		 * @brief Environment light is a special type of light source.
		 * It is not only an ambient and directional light source, but also contributes to environment rendering.
		 * Even being a renderable object, it does not have a solid body.
		*/
		template<bool SM>
		class STPEnvironmentLight;

		template<>
		class STPEnvironmentLight<false> : public STPGlobalLight<false> {
		public:

			STPEnvironmentLight() = default;

			virtual ~STPEnvironmentLight() = default;

			/**
			 * @brief Render the environment.
			*/
			virtual void renderEnvironment() = 0;

		};

		template<>
		class STP_REALISM_API STPEnvironmentLight<true> : public STPGlobalLight<true> {
		protected:

			//The main light source in an environment light is a 
			STPCascadedShadowMap EnvironmentLightShadow;

		public:

			/**
			 * @brief Initialise an environment light instance with shadow map initialised.
			 * @param frustum The pointer to the shadow light frustum for initialising light camera for environment light.
			*/
			STPEnvironmentLight(const STPCascadedShadowMap::STPLightFrustum&);

			virtual ~STPEnvironmentLight() = default;

			const STPLightShadow& getLightShadow() const override;

			/**
			 * @brief Get the pointer to the concrete light shadow instance.
			 * @return The pointer to the concrete implementation shadow instance for environment light.
			*/
			const STPCascadedShadowMap& getEnvironmentLightShadow() const;

		};

	}

}
#endif//_STP_SCENE_LIGHT_H_