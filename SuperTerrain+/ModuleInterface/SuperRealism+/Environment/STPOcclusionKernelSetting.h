#pragma once
#ifndef _STP_OCCLUSION_KERNEL_SETTING_H_
#define _STP_OCCLUSION_KERNEL_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Setting
#include <SuperTerrain+/Environment/STPSetting.hpp>

#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPOcclusionKernelSetting contains properties that determine how the map is sampled for screen-space ambient occlusion.
	*/
	struct STP_REALISM_API STPOcclusionKernelSetting : public STPSetting {
	public:

		//The seed used to generate those random samples
		unsigned long long RandomSampleSeed;
		//Specifies the number of random rotation vector to be used for sampling on the screen.
		//More random vectors give better approximation but may lead to more noisy results.
		//All rotation vectors are stored in a 2D texture and tile onto the rendering screen,
		//such that the size specifies the X, Y dimension of the screen
		glm::uvec2 RotationVectorSize;
		//Specifies the effective sample radius of SSAO.
		float SampleRadius;
		//The bias value helps to resolve some acne effects.
		float Bias;

		STPOcclusionKernelSetting();

		~STPOcclusionKernelSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_OCCLUSION_KERNEL_SETTING_H_