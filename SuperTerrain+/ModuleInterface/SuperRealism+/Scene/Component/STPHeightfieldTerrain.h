#pragma once
#ifndef _STP_HEIGHTFIELD_TERRAIN_H_
#define _STP_HEIGHTFIELD_TERRAIN_H_

#include <SuperRealism+/STPRealismDefine.h>
//Scene Node
#include "../STPSceneObject.hpp"
#include "../../Geometry/STPPlaneGeometry.h"
//GL Utility
#include "../../Object/STPPipelineManager.h"
#include "../../Object/STPBuffer.h"
#include "../../Object/STPBindlessBuffer.h"
#include "../../Object/STPVertexArray.h"
#include "../../Object/STPTexture.h"
#include "../../Object/STPBindlessTexture.h"

//Terrain Generator
#include <SuperTerrain+/World/STPWorldPipeline.h>
#include "../../Environment/STPMeshSetting.h"

//GLM
#include <glm/vec3.hpp>

//System
#include <optional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPHeightfieldTerrain is a real-time photorealistic renderer for heightfield-base terrain.
	*/
	class STP_REALISM_API STPHeightfieldTerrain : public STPSceneObject::STPOpaqueObject {
	public:

		friend class STPWater;

		/**
		 * @brief STPNormalBlendingAlgorithm selects normalmap blending algorithm in the shader
		*/
		enum class STPNormalBlendingAlgorithm : unsigned char {
			//Blend two normalmaps by adding them together
			Linear = 0x00u,
			//Blend by adding x,y component together and multiply z component
			Whiteout = 0x01u,
			//Treat two normalmap as heightmaps and blend based on the height difference
			PartialDerivative = 0x02u,
			//Blend by adding x,y together and use the mesh normal only as z component
			UDN = 0x03u,
			//Create a TBN matrix using mesh normal and transform texture normal to the terrain tangent space
			BasisTransform = 0x04u,
			//Project the texture normal to mesh normal
			RNM = 0x05u,
			//Disable use of texture normalmap, only mesh normal is in effect
			Disable = 0xFFu
		};

		/**
		 * @brief STPTerrainShaderOption specifies the compiler options to be passed to terrain shader compiler.
		*/
		struct STPTerrainShaderOption {
		public:

			//The initial position of the viewer.
			//This is used to initialise the rendered chunk position for the first few frames when heightmap are not yet generated.
			glm::dvec3 InitialViewPosition;

			//Specify the dimension of the noise sampling texture to be used in the shader.
			//Higher scale provides more randomness but also consumes more memory.
			glm::uvec3 NoiseDimension;
			//Set the seed for a texture of random number used during rendering, and regenerate the random texture
			//with the dimension initialised.
			unsigned long long NoiseSeed;

			//Specify the normalmap blending algorithm to be used during rendering.
			STPNormalBlendingAlgorithm NormalBlender;

		};

	private:

		//The main terrain generator
		STPWorldPipeline& TerrainGenerator;
		STPSceneObject::STPDepthRenderGroup::STPLightSpaceDatabase<1u> TerrainDepthRenderer;

		//A buffer representing the terrain plane.
		std::optional<STPPlaneGeometry> TerrainMesh;
		STPBuffer TerrainRenderCommand;

		STPTexture NoiseSample;
		STPBindlessTexture NoiseSampleHandle;

		//Shader program for terrain rendering
		//modeller contains vertex, tes control and tes eval, shader contains geom and frag.
		mutable STPProgramManager TerrainVertex, TerrainModeller, TerrainShader;
		STPPipelineManager TerrainRenderer;

		STPOpenGL::STPint MeshModelLocation, MeshQualityLocation;

		//data for texture splatting
		STPBuffer SplatRegion;
		STPBindlessBuffer SplatRegionAddress;

		/**
		 * @brief Calculate the base chunk position (the coordinate of top-left corner) for the most top-left corner chunk.
		 * @param horizontal_offset The chunk offset in xz direction in world coordinate.
		 * @return The base chunk position.
		*/
		glm::dvec2 calcBaseChunkPosition(glm::dvec2);

		/**
		 * @brief Update the terrain model matrix based on the current centre chunk position.
		*/
		void updateTerrainModel();

	public:

		/**
		 * @brief Initialise the heightfield terrain rendering engine without shadow.
		 * @param generator_pipeline A pointer to the world pipeline that provides heightfield.
		 * @param option The pointer to various compiler options.
		*/
		STPHeightfieldTerrain(STPWorldPipeline&, const STPTerrainShaderOption&);

		STPHeightfieldTerrain(const STPHeightfieldTerrain&) = delete;

		STPHeightfieldTerrain(STPHeightfieldTerrain&&) = delete;

		STPHeightfieldTerrain& operator=(const STPHeightfieldTerrain&) = delete;

		STPHeightfieldTerrain& operator=(STPHeightfieldTerrain&&) = delete;

		~STPHeightfieldTerrain() = default;

		/**
		 * @brief Update the terrain mesh setting.
		 * @param mesh_setting The pointer to the new mesh setting to be updated.
		*/
		void setMesh(const STPEnvironment::STPMeshSetting&);

		/**
		 * @brief Specifically adjust the mesh quality when rendering to depth buffer.
		 * @param tess The pointer to the tessellation setting.
		 * It is recommended to use a (much) lower quality than the actual rendering.
		*/
		void setDepthMeshQuality(const STPEnvironment::STPTessellationSetting&);

		/**
		 * @brief Set the new view position and signal the terrain generator to prepare heightfield texture.
		 * @param viewPos The world position of the viewing coordinate to be prepared.
		*/
		void setViewPosition(const glm::dvec3&);

		bool addDepthConfiguration(size_t, const STPShaderManager*) override;

		/**
		 * @brief Render a regular procedural heightfield terrain.
		 * Terrain texture must be prepared prior to this call, and this function sync with the generator automatically.
		*/
		void render() const override;

		void renderDepth(size_t) const override;

	};

}
#endif//_STP_HEIGHTFIELD_TERRAIN_H_