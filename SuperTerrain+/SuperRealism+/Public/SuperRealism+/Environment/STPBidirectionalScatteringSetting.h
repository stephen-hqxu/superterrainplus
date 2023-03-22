#pragma once
#ifndef _STP_BIDIRECTIONAL_SCATTERING_SETTING_H_
#define _STP_BIDIRECTIONAL_SCATTERING_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPBidirectionalScatteringSetting specifies settings for BSDF rendering function.
	*/
	struct STP_REALISM_API STPBidirectionalScatteringSetting {
	public:

		//Specifies the maximum distance the light ray would travel before hitting a valid object.
		//Increasing ray distance improves quality of intersection test but is more expensive.
		float MaxRayDistance;
		//The bias value applied to depth for testing if the light ray hits the surface.
		//Depth bias helps to reduce some acne artefacts.
		float DepthBias;

		//Specifies the resolution and accuracy of the ray intersection.
		//Ray resolution specifies the how far away the ray travel at each iteration, i.e., how many segment the ray has;
		//a large resolution is faster but may miss small objects.
		//Ray step specifies the number of iteration to precisely search for an intersection point.
		unsigned int RayResolution, RayStep;

		void validate() const;

	};

}
#endif//_STP_BIDIRECTIONAL_SCATTERING_SETTING_H_