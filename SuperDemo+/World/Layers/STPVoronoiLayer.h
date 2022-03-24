#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPVoronoiLayer performs voronoi scaling (4:1 -> 1:1)
	*/
	class STPVoronoiLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	private:

		const bool is3D;

		const Seed voronoi_seed;

		static double sqrtDist(Seed seed, int x, int y, int z, double xFrac, double yFrac, double zFrac) {
			static constexpr auto distribute = [](Seed seed) constexpr -> double {
				const double d = static_cast<double>(static_cast<unsigned int>((seed >> 24ull) % 1024ull)) / 1024.0;
				return (d - 0.5) * 0.9;
			};

			static constexpr auto sqr = [](double n) constexpr -> double {
				return n * n;
			};

			Seed mixed = STPLayer::mixSeed(seed, x);
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

		STPVoronoiLayer(Seed global_seed, Seed salt, bool is3D , STPLayer* parent)
			: STPLayer(global_seed, salt, parent), is3D(is3D), voronoi_seed(std::hash<Seed>{}(global_seed)) {
			
		}

		inline bool is3DScaling() const {
			return this->is3D;
		}

		Sample sample(int x, int y, int z) override {
			//please don't ask me how this algorithm works, because I don't know
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
				ds[c] = STPVoronoiLayer::sqrtDist(this->voronoi_seed, a_abc[0], a_abc[1], a_abc[2], ghs[0], ghs[1], ghs[2]);
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
			return this->getAscendant()->retrieve(xyz[0], this->is3D ? xyz[1] : 0, xyz[2]);
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_