#pragma once
#ifndef _STP_HEIGHTFIELD_TERRAIN_H_
#define _STP_HEIGHTFIELD_TERRAIN_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Utility
#include "../Object/STPPipelineManager.h"
#include "../Object/STPBuffer.h"
#include "../Object/STPVertexArray.h"
#include "../Object/STPTexture.h"
#include "../Utility/STPLogStorage.hpp"

//Terrain Generator
#include <SuperTerrain+/World/STPWorldPipeline.h>
#include "../Environment/STPMeshSetting.h"

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPHeightfieldTerrain is a real-time photorealistic renderer for heightfield-base terrain.
	*/
	class STP_REALISM_API STPHeightfieldTerrain {
	public:

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
			//Contruct a TBN matrix using mesh normal and transform texture normal to the terrain tangent space
			BasisTransform = 0x04u,
			//Project the texture normal to mesh normal
			RNM = 0x05u,
			//Disable use of texture normalmap, only mesh normal is in effect
			Disable = 0xFFu
		};

	private:

		//The main terrain generator
		STPWorldPipeline& TerrainGenerator;

		//A buffer representing the a tile (a unit plane).
		STPBuffer TileBuffer, TileIndex, TerrainRenderCommand;
		STPVertexArray TileArray;
		STPTexture NoiseSample;
		//Shader program for terrain rendering
		STPProgramManager TerrainComponent;
		STPPipelineManager TerrainRenderer;

		std::vector<STPOpenGL::STPuint64> SplatTextureHandle;

		/**
		 * @brief Calculate the base chunk position (the coordinate of top-left corner) for the most top-left corner chunk.
		 * @param horizontal_offset The chunk offset in xz direction in world coordinate.
		 * @return The base chunk position.
		*/
		glm::vec2 calcBaseChunkPosition(const glm::vec2&);

	public:

		typedef STPLogStorage<7ull> STPHeightfieldTerrainLog;

		//The size of the texture storing rangom numbers.
		const glm::uvec3 RandomTextureDimension;

		/**
		 * @brief Initialise the heightfield terrain rendering engine.
		 * @param generator_pipeline A pointer to the world pipeline that provides heightfield.
		 * @param log The pointer to the log output from GL shader and program compiler.
		 * @param noise_scale Specify the dimension of the noise sampling texture to be used in the shader.
		 * Higher scale provides more randomness but also consumes more memory.
		 * @param blending Specify the normalmap blending algorithm to be used during rendering.
		*/
		STPHeightfieldTerrain(STPWorldPipeline&, STPHeightfieldTerrainLog&, glm::uvec3, STPNormalBlendingAlgorithm);

		STPHeightfieldTerrain(const STPHeightfieldTerrain&) = delete;

		STPHeightfieldTerrain(STPHeightfieldTerrain&&) = delete;

		STPHeightfieldTerrain& operator=(const STPHeightfieldTerrain&) = delete;

		STPHeightfieldTerrain& operator=(STPHeightfieldTerrain&&) = delete;

		~STPHeightfieldTerrain();

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
		void setViewPosition(const glm::vec3&);

		/**
		 * @brief Set the terrain light direction.
		 * @param dir The direction of the light.
		*/
		void setLightDirection(const glm::vec3&);

		/**
		 * @brief Render the procedural heightfield terrain.
		 * Terrain texture must be prepared prior to this call, and this function sync with the generator automatically.
		*/
		void operator()() const;

	};

}
#endif//_STP_HEIGHTFIELD_TERRAIN_H_