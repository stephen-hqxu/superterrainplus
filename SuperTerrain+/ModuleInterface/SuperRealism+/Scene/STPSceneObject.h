#pragma once
#ifndef _STP_SCENE_OBJECT_H_
#define _STP_SCENE_OBJECT_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../Object/STPPipelineManager.h"

#include "../Utility/STPShadowInformation.hpp" 

//System
#include <vector>

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
			 * @param light_space_start The starting index to locate the current light space information in the shared memory in shader.
			 * @param light_space_count The number of light space to be captured as a depth texture, as consecutive layered depth textures.
			*/
			virtual void renderDepth(unsigned int, unsigned int) const = 0;

		};

		/**
		 * @brief STPDepthRendererGroup is a utility that helps storing GL program pipeline with different light space count.
		 * This allows compiling one program for each light space count.
		 * Any greater-then-one light space requires layered rendering, a common way is by doing geometry shader instancing.
		 * Yet, geometry shader instancing is configured at shader compile-time.
		 * By grouping depth texture with the same number of layer, this allows choosing a different pipeline and reusing program.
		*/
		class STP_REALISM_API STPDepthRendererGroup {
		private:

			//It is basically a map.
			//Light space size is used for searching for an index, and use this index to locate the pipeline in the other array.
			std::vector<unsigned int> LightSpaceSize;
			std::vector<STPPipelineManager> LightSpaceDepthRenderer;

			/**
			 * @brief Find the first iterator to the configuration table that has configuration no less than the range.
			 * @param light_space_count The number of light space as a key.
			 * @return The iterator to the first element in range of configuration table that is not less than the key.
			*/
			auto getKeyLocation(unsigned int) const;

		public:

			STPDepthRendererGroup() = default;

			STPDepthRendererGroup(const STPDepthRendererGroup&) = delete;

			STPDepthRendererGroup(STPDepthRendererGroup&&) = delete;

			STPDepthRendererGroup& operator=(const STPDepthRendererGroup&) = delete;

			STPDepthRendererGroup& operator=(STPDepthRendererGroup&&) = delete;

			~STPDepthRendererGroup() = default;

			/**
			 * @brief Check if a depth rendering group has been added.
			 * @param light_space_count The group configuration.
			 * @return True if a group with this configuration is found.
			*/
			bool exist(unsigned int) const;

			/**
			 * @brief Add a new rendering pipeline to depth renderer group.
			 * @param light_space_count The number of light space information given to this group.
			 * This will be used as a key to find this group later.
			 * @return The pointer to the newly created group.
			 * Exception is thrown if another group with such key exists.
			*/
			STPPipelineManager& addGroup(unsigned int);

			/**
			 * @brief Find the pipeline for a corresponding light space configuration.
			 * @param light_space_count The number of light space information in the shader.
			 * @return The pointer to the pipeline with given configuration.
			 * Note that for the sake of runtime performance, no error checking is performed.
			 * It is an undefined behaviour if there was no pipeline with such configuration added previously.
			*/
			STPPipelineManager& findGroup(unsigned int);

		};

	}

}
#endif//_STP_SCENE_OBJECT_H_