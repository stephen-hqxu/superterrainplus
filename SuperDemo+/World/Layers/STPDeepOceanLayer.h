#pragma once
#ifndef _STP_DEEP_OCEAN_LAYER_H_
#define _STP_DEEP_OCEAN_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPDeepOceanLayer adds deep ocean if it's surrounded by shallow ocean and not next to land
	*/
	class STPDeepOceanLayer : public STPCrossLayer {
	public:

		STPDeepOceanLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
			//if the centre is not shallow or it's not even ocean, we can't change anything
			if (!Reg::isShallowOcean(center)) {
				return center;
			}

			//test if we are surrounded by shallow ocean
			Sample i = 0;
			if (Reg::isShallowOcean(north)) {
				i++;
			}
			if (Reg::isShallowOcean(east)) {
				i++;
			}
			if (Reg::isShallowOcean(south)) {
				i++;
			}
			if (Reg::isShallowOcean(west)) {
				i++;
			}
			
			//if we are surrounded, generate their relative deep ocean
			if (i > 3) {
				if (center == Reg::Ocean.ID) {
					return Reg::DeepOcean.ID;
				}
				if (center == Reg::WarmOcean.ID) {
					return Reg::DeepWarmOcean.ID;
				}
				if (center == Reg::LukewarmOcean.ID) {
					return Reg::DeepLukewarmOcean.ID;
				}
				if (center == Reg::ColdOcean.ID) {
					return Reg::DeepColdOcean.ID;
				}
				if (center == Reg::FrozenOcean.ID) {
					return Reg::DeepFrozenOcean.ID;
				}

			}

			return center;
		}
	};
}
#endif//_STP_DEEP_OCEAN_LAYER_H_