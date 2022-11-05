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
	namespace STPTextureSplatRuleWrapper {

		//Denotes there is no active region for this rule
		constexpr static unsigned int NoRegion = 0xFFFFFFFFu;

		/**
		 * @brief Get splat registry which contains access information to each splat rule node.
		 * @param splat_db The splatmap database.
		 * @param sample Retrieve the splat registry for this sample.
		 * @return The pointer to the corresponding splat registry. nullptr if no splat registry is found for this
		 * sample.
		*/
		__device__ const STPDiversity::STPTextureInformation::STPSplatRegistry* findSplatRegistry(
			const STPDiversity::STPTextureInformation::STPSplatRuleDatabase&, STPDiversity::Sample);

		/**
		 * @brief Get the active region for this sample and altitude.
		 * @param alt_node The pointer to the altitude node array from the splat database.
		 * @param splat_reg The pointer to the splat registry. If the pointer is null, no region index can be found.
		 * @param alt The altitude to be checked.
		 * @return The region index, which is basically the index to the texture for this active region.
		 * If no region available for this splat registry and rule, NoRegion will be returned.
		*/
		__device__ unsigned int altitudeRegion(const STPDiversity::STPTextureInformation::STPAltitudeNode*,
			const STPDiversity::STPTextureInformation::STPSplatRegistry*, float);

		/**
		 * @brief Get the active region for this sample and gradient.
		 * @param gra_node The pointer to a gradient node array from the splat database.
		 * @param splat_reg The pointer to the splat registry. If the pointer is null, no region index can be found.
		 * @param gra The gradient to be checked.
		 * @param alt The altitude to be checked.
		 * @return The region index, which is basically the index to the texture for this active region.
		 * If no region available for this sample and rule, NoRegion will be returned.
		 * Due to the speciality of gradient rule, it will return the region index to the first region that meets all rules.
		*/
		__device__ unsigned int gradientRegion(const STPDiversity::STPTextureInformation::STPGradientNode*,
			const STPDiversity::STPTextureInformation::STPSplatRegistry*, float, float);

	}

}
#endif//_STP_TEXTURE_SPLAT_RULE_WRAPPER_CUH_