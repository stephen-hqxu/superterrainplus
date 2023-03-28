#pragma once
#ifndef _STP_VORONOI_LAYER_H_
#define _STP_VORONOI_LAYER_H_

#include <functional>

namespace {

	/**
	 * @brief STPVoronoiLayer performs Voronoi scaling (4:1 -> 1:1)
	*/
	class STPVoronoiLayer : public STPLayer {
	private:

		const bool Is3D;

		const STPSeed_t VoronoiSeed;

		static double sqrtDist(const STPSeed_t seed, const int x, const int y, const int z, const double xFrac,
			const double yFrac, const double zFrac) noexcept {
			static constexpr auto distribute = [](const STPSeed_t seed) constexpr noexcept -> double {
				const double d = static_cast<double>(static_cast<unsigned int>((seed >> 24ull) % 1024ull)) / 1024.0;
				return (d - 0.5) * 0.9;
			};

			static constexpr auto sqr = [](const double n) constexpr noexcept -> double {
				return n * n;
			};

			STPSeed_t mixed = STPLayer::mixSeed(seed, x);
			mixed = STPLayer::mixSeed(mixed, y);
			mixed = STPLayer::mixSeed(mixed, z);
			mixed = STPLayer::mixSeed(mixed, x);
			mixed = STPLayer::mixSeed(mixed, y);
			mixed = STPLayer::mixSeed(mixed, z);

			const double d = distribute(mixed);
			mixed = STPLayer::mixSeed(mixed, seed);
			const double e = distribute(mixed);
			mixed = STPLayer::mixSeed(mixed, seed);
			const double f = distribute(mixed);

			return sqr(zFrac + f) + sqr(yFrac + e) + sqr(xFrac + d);
		}

	public:

		STPVoronoiLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, const bool is3D, STPLayer& parent) :
			STPLayer(cache_size, global_seed, salt, { parent }),
			Is3D(is3D), VoronoiSeed(std::hash<STPSeed_t> {}(global_seed)) {
			
		}

		STPSample_t sample(const int x, const int y, const int z) override {
			const int ijk[3] = {
				x - 2, 
				y - 2, 
				z - 2
			};
			const int lmn[3] = {
				ijk[0] >> 2, 
				ijk[1] >> 2, 
				ijk[2] >> 2 
			};
			const double def[3] = {
				static_cast<double>((ijk[0] & 3)) / 4.0, 
				static_cast<double>((ijk[1] & 3)) / 4.0, 
				static_cast<double>((ijk[2] & 3)) / 4.0 
			};
			double ds[8];

			for (unsigned int c = 0u; c < 8u; c++) {
				const bool bl[3] = {(c & 4u) == 0u, (c & 2u) == 0u, (c & 1u) == 0u};
				const int a_abc[3] = {
					bl[0] ? lmn[0] : lmn[0] + 1,
					bl[1] ? lmn[1] : lmn[1] + 1,
					bl[2] ? lmn[2] : lmn[2] + 1
				};
				const double ghs[3] = {
					bl[0] ? def[0] : def[0]  - 1.0,
					bl[1] ? def[1] : def[1] - 1.0,
					bl[2] ? def[2] : def[2] - 1.0
				};
				ds[c] = STPVoronoiLayer::sqrtDist(this->VoronoiSeed, a_abc[0], a_abc[1], a_abc[2], ghs[0], ghs[1], ghs[2]);
			}

			unsigned int index = 0u;
			double min = ds[0];

			for (unsigned int c = 1u; c < 8u; c++) {
				if (ds[c] >= min) {
					continue;
				}
				index = c;
				min = ds[c];
			}

			const int xyz[3] = {
				(index & 4u) == 0u ? lmn[0] : lmn[0] + 1,
				(index & 2u) == 0u ? lmn[1] : lmn[1] + 1,
				(index & 1u) == 0u ? lmn[2] : lmn[2] + 1
			};
			return this->getAscendant().retrieve(xyz[0], this->Is3D ? xyz[1] : 0, xyz[2]);
		}

	};

}
#endif//_STP_VORONOI_LAYER_H_