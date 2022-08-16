#pragma once
#ifndef _STP_EXTENDED_SCENE_PIPELINE_H_
#define _STP_EXTENDED_SCENE_PIPELINE_H_

#include <SuperRealism+/STPRealismDefine.h>
//OptiX
#include <optix_types.h>

//System
#include <type_traits>
#include <memory>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPExtendedScenePipeline is an extension on STPScenePipeline.
	 * The traditional rendering pipeline provides basic rendering functionality with rasterisation.
	 * Due to limited ability of such rendering technique when comes to ultimate visual quality,
	 * the extended rendering pipeline introduces global illumination.
	 * This new scene pipeline allows combination of the vanilla scene pipeline to achieve hybrid rendering.
	 * Similar to STPScenePipeline, it is recommended to create only one STPExtendedScenePipeline instance per program.
	 * @see STPScenePipeline
	*/
	class STP_REALISM_API STPExtendedScenePipeline {
	private:

		/**
		 * @brief STPDeviceContextDestroyer destroys the OptiX device context.
		*/
		struct STPDeviceContextDestroyer {
		public:

			void operator()(OptixDeviceContext) const;

		};
		//For simplicity of management, each instance of scene pipeline holds a context
		std::unique_ptr<std::remove_pointer_t<OptixDeviceContext>, STPDeviceContextDestroyer> Context;

		/**
		 * @brief A simple utility that launches a ray from screen space and reports intersection.
		 * It also provides geometry data where the ray intersects, this is useful for rendering mirror reflection.
		 * The ray terminates at the closest hit, or missed.
		*/
		class STPScreenSpaceRayIntersection;
		std::unique_ptr<STPScreenSpaceRayIntersection> IntersectionTracer;

	public:

		STPExtendedScenePipeline();

		STPExtendedScenePipeline(const STPExtendedScenePipeline&) = delete;

		STPExtendedScenePipeline(STPExtendedScenePipeline&&) = delete;

		STPExtendedScenePipeline& operator=(const STPExtendedScenePipeline&) = delete;

		STPExtendedScenePipeline& operator=(STPExtendedScenePipeline&&) = delete;

		~STPExtendedScenePipeline();
	
	};

}
#endif//_STP_EXTENDED_SCENE_PIPELINE_H_