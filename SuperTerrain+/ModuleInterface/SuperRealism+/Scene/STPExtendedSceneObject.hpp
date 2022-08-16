#pragma once
#ifndef _STP_EXTENDED_SCENE_OBJECT_HPP_
#define _STP_EXTENDED_SCENE_OBJECT_HPP_

#include <optix.h>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPExtendedSceneObject is an extension from STPSceneObject,
	 * and is a collection of all different renderable objects in an extended scene.
	 * @see STPSceneObject
	*/
	namespace STPExtendedSceneObject {

		/**
		 * @brief STPTraceable is a base class for an object that can be rendered via ray tracing.
		*/
		class STPTraceable {
		public:

			STPTraceable() = default;

			virtual ~STPTraceable() = default;

		};

	}

}
#endif//_STP_EXTENDED_SCENE_OBJECT_HPP_