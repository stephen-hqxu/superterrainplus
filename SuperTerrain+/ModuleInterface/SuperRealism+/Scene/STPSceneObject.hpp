#pragma once
#ifndef _STP_SCENE_OBJECT_HPP_
#define _STP_SCENE_OBJECT_HPP_

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
		*/
		class STPOpaqueObject : public STPRenderable {
		public:

			/**
			 * @brief Initialise a new opaque object.
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

			/**
			 * @brief Check if the current environment object is visible based on its current visibility value.
			 * @return True if the object has visibility greater than zero, false otherwise.
			*/
			bool isEnvironmentVisible() const {
				return this->EnvironmentVisibility > 0.0f;
			}

		};

		/**
		 * @brief STPAnimatedObject is a special type of object that carries procedural animation.
		 * Their animation can be controlled explicitly by application using a timer.
		*/
		class STPAnimatedObject {
		public:

			/**
			 * @brief Init a new STPAnimatedObject.
			*/
			STPAnimatedObject() = default;

			virtual ~STPAnimatedObject() = default;

			/**
			 * @brief Update the animation timer for the next animation frame.
			 * @param second The absolute animation time, starting at zero, in second for the new animation frame.
			 * This time can be, for example, the time elapsed since the start of the program.
			*/
			virtual void updateAnimationTimer(double) = 0;

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
		namespace STPDepthRenderGroup {
			//A collection of all depth shaders program in a group.
			template<size_t GS>
			using STPShaderCollection = std::array<STPProgramManager, GS>;
			//All members in a depth group.
			template<size_t GS>
			using STPGroupMember = std::pair<STPPipelineManager, STPShaderCollection<GS>>;

			//Light space size is used for searching for an index,
			//and use this index to locate the pipeline in the other array.
			template<size_t GS>
			using STPLightSpaceDatabase = std::unordered_map<size_t, STPGroupMember<GS>>;
		}

	}

}
#endif//_STP_SCENE_OBJECT_HPP_