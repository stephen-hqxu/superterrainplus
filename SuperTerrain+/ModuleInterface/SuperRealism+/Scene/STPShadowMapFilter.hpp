#pragma once
#ifndef _STP_SHADOW_MAP_FILTER_HPP_
#define _STP_SHADOW_MAP_FILTER_HPP_

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPShadowMapFilter defines filtering technologies used for post-process shadow maps.
	 * Underlying value greater than VSM denotes VSM-derived shadow map filters.
	*/
	enum class STPShadowMapFilter : unsigned char {
		//Nearest-Neighbour filter, shadow value is read from the nearest pixel.
		Nearest = 0x00u,
		//Bilinear filter, shadow value is read from its neighbours and linearly interpolated.
		Bilinear = 0x01u,
		//Percentage-Closer filter, it attempts to smooth the edge of the shadow using a blur kernel.
		PCF = 0x02u,
		//Stratified Sampled PCF, this is a variant of PCF which uses random stratified sampling when convolving the kernel.
		SSPCF = 0x03u,
		//Variance Shadow Mapping, it uses variance to estimate the likelihood of a pixel that should have shadow 
		//after having the shadow map blurred.
		VSM = 0x10u,
		//Exponential Shadow Mapping, it is a derivation of VSM. Instead of using Chebyshev's inequality to approximate the probability,
		//an exponential function is used.
		ESM = 0x11u
	};

}
#endif//_STP_SHADOW_MAP_FILTER_HPP_