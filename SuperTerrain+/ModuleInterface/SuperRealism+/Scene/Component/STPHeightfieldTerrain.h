#pragma once
#ifndef _STP_HEIGHTFIELD_TERRAIN_H_
#define _STP_HEIGHTFIELD_TERRAIN_H_

#include <SuperRealism+/STPRealismDefine.h>
//Scene Node
#include "../STPSceneObject.h"
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
#include <queue>
#include <list>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPHeightfieldTerrain is a real-time photorealistic renderer for heightfield-base terrain.
	 * @tparam SM Indicate if the terrain renderer should allow render to shadow maps.
	*/
	template<bool SM>
	class STPHeightfieldTerrain;

	template<>
	class STP_REALISM_API STPHeightfieldTerrain<false> : public STPSceneObject::STPOpaqueObject<false> {
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

			//Specify the dimension of the noise sampling texture to be used in the shader.
			//Higher scale provides more randomness but also consumes more memory.
			glm::uvec3 NoiseDimension;
			//Specify the normalmap blending algorithm to be used during rendering.
			STPNormalBlendingAlgorithm NormalBlender;

		};

	protected:

		//The main terrain generator
		STPWorldPipeline& TerrainGenerator;

		//A buffer representing the terrain plane.
		std::optional<STPPlaneGeometry> TerrainMesh;
		STPBuffer TerrainRenderCommand;

		STPTexture NoiseSample;
		std::optional<STPBindlessTexture> NoiseSampleHandle;

		//Shader program for terrain rendering
		//modeller contains vertex, tes control and tes eval, shader contains geom and frag.
		mutable STPProgramManager TerrainVertex, TerrainModeller, TerrainShader;
		STPPipelineManager TerrainRenderer;

		STPOpenGL::STPint MeshModelLocation;

		//data for texture splatting
		STPBuffer SplatRegion;
		std::optional<STPBindlessBuffer> SplatRegionAddress;
		std::vector<STPBindlessTexture> SplatTextureHandle;

		/**
		 * @brief Calculate the base chunk position (the coordinate of top-left corner) for the most top-left corner chunk.
		 * @param horizontal_offset The chunk offset in xz direction in world coordinate.
		 * @return The base chunk position.
		*/
		glm::dvec2 calcBaseChunkPosition(glm::dvec2);

	public:

		//The size of the texture storing random numbers.
		const glm::uvec3 RandomTextureDimension;

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

		virtual ~STPHeightfieldTerrain() = default;

		/**
		 * @brief Update the terrain mesh setting.
		 * @param mesh_setting The pointer to the new mesh setting to be updated.
		*/
		void setMesh(const STPEnvironment::STPMeshSetting&);

		/**
		 * @brief Set the seed for a texture of random number used during rendering, and regenerate the random texture 
		 * with the dimension initialised.
		 * This function is considered to be expensive and hence should not be called frequently.
		 * @param seed The new seed value.
		*/
		void seedRandomBuffer(unsigned long long);

		/**
		 * @brief Set the new view position and signal the terrain generator to prepare heightfield texture.
		 * @param viewPos The world position of the viewing coordinate to be prepared.
		*/
		void setViewPosition(const glm::dvec3&);

		/**
		 * @brief Render a regular procedural heightfield terrain.
		 * Terrain texture must be prepared prior to this call, and this function sync with the generator automatically.
		*/
		void render() const override;

	};

	template<>
	class STP_REALISM_API STPHeightfieldTerrain<true> : public STPSceneObject::STPOpaqueObject<true>, public STPHeightfieldTerrain<false> {
	private:

		//depth renderer prunes the frag shader.
		STPSceneObject::STPDepthRenderGroup<1ull> TerrainDepthRenderer;

		const STPOpenGL::STPint MeshQualityLocation;

	public:

		/**
		 * @brief Initialise the heightfield terrain rendering engine with shadow rendering.
		 * Arguments are exactly the same as the base version.
		 * @see STPHeightfieldTerrain<false>.
		*/
		STPHeightfieldTerrain(STPWorldPipeline&, const STPTerrainShaderOption&);

		~STPHeightfieldTerrain() = default;

		/**
		 * @brief Specifically adjust the mesh quality when rendering to depth buffer.
		 * @param tess The pointer to the tessellation setting.
		 * It is recommended to use a (much) lower quality than the actual rendering.
		*/
		void setDepthMeshQuality(const STPEnvironment::STPTessellationSetting&);

		bool addDepthConfiguration(size_t, const STPShaderManager*) override;

		void renderDepth(size_t) const override;

	};

}
#endif//_STP_HEIGHTFIELD_TERRAIN_H_