#pragma once
#ifndef _STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_
#define _STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_

//Texture Splat Rule
#include <SuperTerrain+/World/Diversity/Texture/STPTextureInformation.hpp>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPTextureSplatRuleWrapper is a thin wrapper to STPTextureInformation and accesses data within to generate
	 * rule-based biome-dependent texture splatmap.
	*/
	struct STPTextureSplatRuleWrapper {
	public:

		const STPDiversity::STPTextureInformation::STPSplatRuleDatabase& SplatRule;

		//Denotes there is no active region for this rule
		constexpr static unsigned int NoRegion = 0xFFFFFFFFu;

		/**
		 * @brief Initialise a texture splat rule wrapper.
		 * @param splat_db The pointer to the splat rule database to access.
		*/
		__device__ STPTextureSplatRuleWrapper(const STPDiversity::STPTextureInformation::STPSplatRuleDatabase&);

		__device__ ~STPTextureSplatRuleWrapper();

		/**
		 * @brief Get splat registry which contains access information to each splat rule node.
		 * @param sample Retrieve the splat registry for this sample.
		 * @return The pointer to the corresponding splat registry. nullptr if no splat registry is found for this
		 * sample.
		*/
		__device__ const STPDiversity::STPTextureInformation::STPSplatRegistry* findSplatRegistry(STPSample_t) const;

		/**
		 * @brief Get the active region for this sample and altitude.
		 * @param splat_reg The pointer to the splat registry. If the pointer is null, no region index can be found.
		 * @param alt The altitude to be checked.
		 * @return The region index, which is basically the index to the texture for this active region.
		 * If no region available for this splat registry and rule, NoRegion will be returned.
		*/
		__device__ unsigned int altitudeRegion(const STPDiversity::STPTextureInformation::STPSplatRegistry*, float) const;

		/**
		 * @brief Get the active region for this sample and gradient.
		 * @param splat_reg The pointer to the splat registry. If the pointer is null, no region index can be found.
		 * @param gra The gradient to be checked.
		 * @param alt The altitude to be checked.
		 * @return The region index, which is basically the index to the texture for this active region.
		 * If no region available for this sample and rule, NoRegion will be returned.
		 * Due to the speciality of gradient rule, it will return the region index to the first region that meets all rules.
		*/
		__device__ unsigned int gradientRegion(const STPDiversity::STPTextureInformation::STPSplatRegistry*, float, float) const;

	};

}
#endif//_STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_