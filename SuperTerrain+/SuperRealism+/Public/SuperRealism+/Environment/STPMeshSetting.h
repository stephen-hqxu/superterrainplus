#pragma once
#ifndef _STP_MESH_SETTING_H_
#define _STP_MESH_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Setting
#include "STPTessellationSetting.h"

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPMeshSettings stores parameters for rendering terrain mesh
	*/
	struct STP_REALISM_API STPMeshSetting {
	public:

		/**
		 * @brief STPTextureRegionSmoothSetting controls parameters for terrain texture splatting region smoothing algorithms.
		 * The algorithm blends colours from different regions.
		*/
		struct STP_REALISM_API STPTextureRegionSmoothSetting {
		public:

			//Control the convolution radius for the box blur filter.
			//Higher radius gives smoother result but consumes more power.
			unsigned int KernelRadius;
			//Control the distance between each sampling points.
			float KernelScale;
			//The UV multiplier applied to noise texture.
			//Lower value gives smoother noise.
			unsigned int NoiseScale;

			void validate() const;

		};

		/**
		 * @brief STPTextureScaleDistanceSetting specifies how the system should use the multi-scale texture blending.
		*/
		struct STP_REALISM_API STPTextureScaleDistanceSetting {
		public:

			//The i-th far specifies the minimum distance factor from viewer to texel to enable N-th texture scale, otherwise (i+1)-th scale is used.
			//This distance factor is a multiple to the maximum viewing distance.
			//If none of the distance is satisfied, the N-th scale is used where N is the number scale settings in total.
			//Except for the first distance which will always use the first scale, 
			//and the texel distances outside the last far distance which will always use the last scale,
			//any other i-th distance makes the texture being blended with texture scaled by the (i-1)-th.
			float PrimaryFar, SecondaryFar, TertiaryFar;

			void validate() const;

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
		 * @brief Determine the range of tessellations, and the tessellation LoD will be clamped between min and max within a specified range of distances.
		*/
		STPTessellationSetting TessSetting;

		//Fragment Control
		STPTextureScaleDistanceSetting RegionScaleSetting;
		STPTextureRegionSmoothSetting RegionSmoothSetting;

		void validate() const;

	};

}
#endif//_STP_MESH_SETTING_HPP_