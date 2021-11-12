#pragma once
#ifndef __CUDACC__
#error __FILE__ can only be compiled by nvcc and nvrtc exclusively
#endif

#ifndef _STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_
#define _STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif//__CUDACC_RTC__

//Texture Splat Rule
#include <SuperTerrain+/World/Diversity/Texture/STPTextureInformation.hpp>

#include <limits>

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPTextureSplatRuleWrapper is a thin wrapper to STPTextureInformation and accesses data within to generate
	 * rule-based biome-dependent texture splatmap.
	*/
	class STPTextureSplatRuleWrapper {
	private:

		const STPDiversity::STPTextureInformation::STPSplatRuleDatabase& SplatRule;

		/**
		 * @brief Get splat registry which contains access information to each splat rule node
		 * @param sample Retrieve the spalt registry for this sample.
		 * @return The pointer to the corresponding splat registry. nullptr if no splat registry is found for this sample
		*/
		__device__ const STPDiversity::STPTextureInformation::STPSplatRegistry* getSplatRegistry(STPDiversity::Sample) const;

	public:
	
		//Denotes there is no active region for this rule
		constexpr static unsigned int NoRegion = std::numeric_limits<unsigned int>::max();

		/**
		 * @brief Initialise a texture splat rule wrapper
		 * @param splat_database The pointer to a collection of all splat rules
		*/
		__device__ STPTextureSplatRuleWrapper(const STPDiversity::STPTextureInformation::STPSplatRuleDatabase&);

		__device__ STPTextureSplatRuleWrapper(const STPTextureSplatRuleWrapper&) = delete;

		__device__ STPTextureSplatRuleWrapper(STPTextureSplatRuleWrapper&&) = delete;

		__device__ STPTextureSplatRuleWrapper& operator=(const STPTextureSplatRuleWrapper&) = delete;

		__device__ STPTextureSplatRuleWrapper& operator=(STPTextureSplatRuleWrapper&&) = delete;

		__device__ ~STPTextureSplatRuleWrapper();
		
		/**
		 * @brief Get the active region for this sample and altitude.
		 * @param sample The active region from which sample will be used
		 * @param alt The altitude to be checked
		 * @return The region index, which is basically the index to the texture for this active region.
		 * If no region available for this sample and rule, NoRegion will be returned
		*/
		__device__ unsigned int altitudeRegion(STPDiversity::Sample, float) const;

		/**
		 * @brief Get the active region for this sample and gradient
		 * @param sample The active region from which sample will be used
		 * @param gra The gradient to be checked
		 * @param alt The altitude to be checked
		 * @return The region index, which is basically the index to the texture for this active region.
		 * If no region avilable for this sample and rule, NoRegion will be returned.
		 * Due to the speciality of gradient rule, it will return the region index to the first region that meets all rules.
		*/
		__device__ unsigned int gradientRegion(STPDiversity::Sample, float, float) const;

	};

}
#endif//_STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_