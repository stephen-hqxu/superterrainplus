#pragma once
#ifndef _STP_SCENE_OBJECT_H_
#define _STP_SCENE_OBJECT_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../Object/STPShaderManager.h"
#include "../Object/STPProgramManager.h"
#include "../Object/STPPipelineManager.h"

//Container
#include <array>
#include <unordered_map>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSceneObject is a collection of all different renderable objects in a scene.
	*/
	namespace STPSceneObject {

		/**
		 * @brief STPRenderable is a base class for all objects that can be rendered.
		*/
		class STPRenderable {
		public:

			/**
			 * @brief Initialise the renderable object.
			*/
			STPRenderable() = default;

			virtual ~STPRenderable() = default;

			/**
			 * @brief Render the object.
			*/
			virtual void render() const = 0;

		};

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
		class STPOpaqueObject<false> : public STPRenderable {
		public:

			/**
			 * @brief Initialise a new opaque object that does not cast shadow.
			*/
			STPOpaqueObject() = default;

			virtual ~STPOpaqueObject() = default;

		};

		template<>
		class STPOpaqueObject<true> {
		public:

			/**
			 * @brief Initialise a new opaque object that casts shadow.
			*/
			STPOpaqueObject() = default;

			virtual ~STPOpaqueObject() = default;

			/**
			 * @brief Add a new depth configuration.
			 * A depth configuration can be considered as a group of shadow-casting light sources with the same light parameters.
			 * @param light_space_count The number of light space to be captured as a depth texture, as consecutive layered depth textures.
			 * @param depth_shader A fragment shader for processing additional operations during depth rendering.
			 * The depth shader can be optionally null to denote there is no depth shader to be used.
			 * @return The implementation should guarantee the depth configuration is unique in an opaque object,
			 * therefore if this depth configuration has been added previously, no operation should be performed and false should be returned.
			 * If a new configuration is added, returns true.
			*/
			virtual bool addDepthConfiguration(size_t, const STPShaderManager*) = 0;

			/**
			 * @brief Render the opaque object to a depth texture, with shading pruned.
			 * @param light_space_count The number of light space to be captured as a depth texture, as consecutive layered depth textures.
			*/
			virtual void renderDepth(size_t) const = 0;

		};

		/**
		 * @brief STPTransparentObject is a type of object that allows light to partially or completely pass through.
		 * It supports rendering of reflective and refractive materials. Transparent object currently does not support shadow casting.
		*/
		class STPTransparentObject : public STPRenderable {
		public:

			/**
			 * @brief Initialise a new transparent object.
			*/
			STPTransparentObject() = default;

			virtual ~STPTransparentObject() = default;

		};

		/**
		 * @brief STPEnvironmentObject is a special type of object that contributes to the environmental effects.
		 * It does not have a solid body and does not interact with other objects in the scene.
		*/
		class STPEnvironmentObject : public STPRenderable {
		public:

			//Specifies an output colour intensity multiplier.
			//Set this value to zero or any negative value will effectively skips this environment object during rendering.
			//Setting to a value greater than one will be clamped.
			float EnvironmentVisibility = 1.0f;

			/**
			 * @brief Init a new STPEnvironmentObject.
			*/
			STPEnvironmentObject() = default;

			virtual ~STPEnvironmentObject() = default;

		};

		/**
		 * @brief STPDepthRenderGroup is a utility that helps storing GL program pipeline with different light space count.
		 * This allows compiling one program for each light space count.
		 * Any greater-then-one light space requires layered rendering, a common way is by doing geometry shader instancing.
		 * Yet, geometry shader instancing is configured at shader compile-time.
		 * By grouping depth texture with the same number of layer, this allows choosing a different pipeline and reusing program.
		 * @param GS Group size. Specifies how many specialised shader program should exist in a rendering pipeline, therefore group size.
		 * As GL pipeline object allows free-combination of different programs, so it is recommended to reuse shader program 
		 * and only create those that change based on different depth rendering configuration.
		*/
		template<size_t GS>
		class STP_REALISM_API STPDepthRenderGroup {
		public:

			static_assert(GS <= 2ull, "Depth render group currently only supports group size up to 2");

			//Contains all depth shader program in a group
			typedef std::array<STPProgramManager, GS> STPShaderCollection;
			//All members in a depth group
			typedef std::pair<STPPipelineManager, STPShaderCollection> STPGroupMember;

		private:

			//Light space size is used for searching for an index, and use this index to locate the pipeline in the other array.
			std::unordered_map<size_t, STPGroupMember> LightSpaceDatabase;

		public:

			STPDepthRenderGroup() = default;

			STPDepthRenderGroup(const STPDepthRenderGroup&) = delete;

			STPDepthRenderGroup(STPDepthRenderGroup&&) = delete;

			STPDepthRenderGroup& operator=(const STPDepthRenderGroup&) = delete;

			STPDepthRenderGroup& operator=(STPDepthRenderGroup&&) = delete;

			~STPDepthRenderGroup() = default;

			/**
			 * @brief Check if a depth rendering group has been added.
			 * @param light_space_count The group configuration.
			 * @return True if a group with this configuration is found.
			*/
			bool exist(size_t) const;

			/**
			 * @brief Add a new rendering pipeline to depth renderer group.
			 * @param light_space_count The number of light space information given to this group.
			 * This will be used as a key to find this group later.
			 * @return A pointer to a pair of pointers to an array of shader program and a program pipeline.
			 * It is recommended that the implementation uses the given program shader to create the pipeline.
			 * All pointers returned is guaranteed to be valid until the end of life of the depth render group instance.
			 * Exception is thrown if another group with such key exists.
			*/
			STPGroupMember& addGroup(size_t);

			/**
			 * @see The constant version of this function
			*/
			STPPipelineManager& findPipeline(size_t);

			/**
			 * @brief Find the pipeline for a corresponding light space configuration.
			 * @param light_space_count The number of light space information in the shader.
			 * @return The pointer to the pipeline with given configuration.
			 * It is an undefined behaviour if no such pipeline exists, to avoid expensive runtime check.
			*/
			const STPPipelineManager& findPipeline(size_t) const;

		};

	}

}
#endif//_STP_SCENE_OBJECT_H_