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
	private:

		//The main terrain generator
		STPWorldPipeline& TerrainGenerator;

		//A buffer representing the a tile (a unit plane).
		STPBuffer TileBuffer, TileIndex, TerrainRenderCommand;
		STPVertexArray TileArray;
		STPTexture NoiseSample;
		//Shader program for terrain rendering
		mutable STPProgramManager TerrainComponent;
		STPPipelineManager TerrainRenderer;

		std::vector<STPOpenGL::STPuint64> SplatTextureHandle;

		//The last identified view position, acting as a cache
		glm::vec3 ViewPosition;

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
		*/
		STPHeightfieldTerrain(STPWorldPipeline&, STPHeightfieldTerrainLog&, glm::uvec3);

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
		 * @brief Signal the terrain generator to prepare heightfield texture before actual rendering.
		 * @param viewPos The world position of the viewing coordinate to be prepared.
		*/
		void prepare(const glm::vec3&);

		/**
		 * @brief Render the procedural heightfield terrain.
		 * Terrain texture must be prepared prior to this call, and this function sync with the generator automatically.
		*/
		void operator()() const;

	};

}
#endif//_STP_HEIGHTFIELD_TERRAIN_H_