#pragma once
#ifndef _STP_SCENE_OBJECT_HPP_
#define _STP_SCENE_OBJECT_HPP_

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSceneObject is a collection of all different renderable objects in a scene.
	*/
	namespace STPSceneObject {

		/**
		 * @brief STPOpaqueObject is the base of all opaque objects.
		 * This type of object does not allow light to pass through and can optionally cast shadow.
		 * @tparam SM True to denote this opaque object should cast a shadow,
		 * meaning the depth information of this object will be rendered to a shadow map.
		 * Usually, an opaque object implementation using shadow may simply derived from the non-casting one.
		*/
		template<bool SM>
		class STPOpaqueObject;

		template<>
		class STPOpaqueObject<false> {
		public:

			/**
			 * @brief Initialise a new opaque object that does not cast shadow.
			*/
			STPOpaqueObject() = default;

			virtual ~STPOpaqueObject() = default;

			/**
			 * @brief Render the opaque object.
			*/
			virtual void render() const = 0;

		};

		//To avoid running into the diamond problem, the shadow-casting opaque object does not inherit from the non-casting one.
		template<>
		class STPOpaqueObject<true> {
		public:

			/**
			 * @brief Initialise a new opaque object that cast shadow.
			*/
			STPOpaqueObject() = default;

			virtual ~STPOpaqueObject() = default;

			/**
			 * @brief Render the opaque object to a depth texture, with shading pruned.
			*/
			virtual void renderDepth() const = 0;

		};

	}

}
#endif//_STP_SCENE_OBJECT_HPP_