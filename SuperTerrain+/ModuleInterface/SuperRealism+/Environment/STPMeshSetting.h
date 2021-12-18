#pragma once
#ifndef _STP_MESH_SETTING_H_
#define _STP_MESH_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Setting
#include <SuperTerrain+/Environment/STPSetting.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPMeshSettings stores parameters for rendering terrain mesh
	*/
	struct STP_REALISM_API STPMeshSetting : public STPSetting {
	public:

		/**
		 * @brief STPTessellationSettings controls the range of the tessellation levels, as well as the min and max distance
		 * where tessellation will become min and max
		*/
		struct STP_REALISM_API STPTessellationSetting : public STPSetting {
		public:

			/**
			 * @brief Determine the maximum tessellation level when the distance falls beyond FurthestTessDistance
			*/
			float MaxTessLevel;

			/**
			 * @brief Determine the minumum tessellation level when the distance falls below NearestTessDistance
			*/
			float MinTessLevel;

			/**
			 * @brief Determine the maximum tessellation distance where tess level beyong will be clamped to MaxTessLevel
			*/
			float FurthestTessDistance;

			/**
			 * @brief Determine the minimum tessellation distance where tess level below will be clamped to MinTessLevel
			*/
			float NearestTessDistance;

			/**
			 * @brief Init STPTessellationSettings with defaults
			*/
			STPTessellationSetting();

			~STPTessellationSetting() = default;

			bool validate() const override;

		};

		/**
		 * @brief STPTextureRegionSmoothSetting controls parameters for terrain texture splatting region smoothing algorithms.
		 * The algorithm blends colors from different regions.
		*/
		struct STP_REALISM_API STPTextureRegionSmoothSetting : public STPSetting {
		public:

			//Control the convolution radius for the box blur filter.
			//Higher radius gives smoother result but consumes more power.
			unsigned int KernelRadius;
			//Control the distance between each sampling points.
			float KernelScale;
			//The UV multiplier applied to noise texture.
			//Lower value gives smoother noise.
			float NoiseScale;

			/**
			 * @brief Init STPTextureRegionSmoothSetting with default settings.
			*/
			STPTextureRegionSmoothSetting();

			~STPTextureRegionSmoothSetting() = default;

			bool validate() const override;

		};

		//Mesh Normal Parameters
		//Control the strength of z component of the normal map, the greater, the more the normal pointing towards the surface
		float Strength;

		//Tessellation Control
		/**
		 * @brief Specify the height multiplier on the heightmap
		*/
		float Altitude;
		/**
		 * @brief Determine how far the mesh starts to decrease its LoD, (0, inf), in classic hermite interpolation, this factor will be 8.0f
		 * 2.0 is the default value, mesh will half its original LoD at 50% of tessllation distance
		*/
		float LoDShiftFactor;
		/**
		 * @brief Determine the range of tessellations, and the tessellation LoD will be clamped between min and max within a specified range of distances.
		*/
		STPTessellationSetting TessSetting;

		//Fragment Control
		STPTextureRegionSmoothSetting RegionSmoothSetting;
		//Texture coordinate scaling control
		//Texture coordinate will be calculated like this:
		//UV * (Factor * TextureDimension ^ -1)
		//The result should be a multiple of number of rendered chunk to avoid artifacts.
		unsigned int UVScaleFactor;

		/**
		 * @brief Init STPMeshSettings with defaults
		*/
		STPMeshSetting();

		~STPMeshSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_MESH_SETTING_HPP_