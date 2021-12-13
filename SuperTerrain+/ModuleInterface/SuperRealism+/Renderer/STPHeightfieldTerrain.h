#pragma once
#ifndef _STP_HEIGHTFIELD_TERRAIN_H_
#define _STP_HEIGHTFIELD_TERRAIN_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Utility
#include "../Object/STPPipelineManager.h"
#include "../Object/STPBuffer.h"
#include "../Object/STPVertexArray.h"
#include "../Utility/STPLogStorage.hpp"

//Terrain Generator
#include <SuperTerrain+/World/STPWorldPipeline.h>
#include <SuperTerrain+/Environment/STPMeshSetting.h>

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
		//Shader program for terrain rendering
		mutable STPProgramManager TerrainComponent;
		STPPipelineManager TerrainRenderer;

		std::vector<STPOpenGL::STPuint64> SplatTextureHandle;

		/**
		 * @brief Calculate the base chunk position (the coordinate of top-left corner) for the most top-left corner chunk.
		 * @param horizontal_offset The chunk offset in xz direction in world coordinate.
		 * @return The base chunk position.
		*/
		glm::vec2 calcBaseChunkPosition(const glm::vec2&);

	public:

		typedef STPLogStorage<8ull> STPHeightfieldTerrainLog;

		/**
		 * @brief Initialise the heightfield terrain rendering engine.
		 * @param generator_pipeline A pointer to the world pipeline that provides heightfield.
		 * @param log The pointer to the log output from GL shader and program compiler.
		*/
		STPHeightfieldTerrain(STPWorldPipeline&, STPHeightfieldTerrainLog&);

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
		 * @brief Render the procedural heightfield terrain.
		 * @param viewPos The world position of the viewing coordinate.
		*/
		void operator()(const glm::vec3&) const;

	};

}
#endif//_STP_HEIGHTFIELD_TERRAIN_H_