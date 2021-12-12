#pragma once
#ifndef _STP_HEIGHTFIELD_TERRAIN_H_
#define _STP_HEIGHTFIELD_TERRAIN_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Utility
#include "../Object/STPPipelineManager.h"
#include "../Object/STPBuffer.h"
#include "../Object/STPVertexArray.h"

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
		STPProgramManager TerrainComponent;
		STPPipelineManager TerrainRenderer;

	public:

		/**
		 * @brief Initialise the heightfield terrain rendering engine.
		 * @param generator_pipeline A pointer to the world pipeline that provides heightfield.
		*/
		STPHeightfieldTerrain(STPWorldPipeline&);

		STPHeightfieldTerrain(const STPHeightfieldTerrain&) = delete;

		STPHeightfieldTerrain(STPHeightfieldTerrain&&) = delete;

		STPHeightfieldTerrain& operator=(const STPHeightfieldTerrain&) = delete;

		STPHeightfieldTerrain& operator=(STPHeightfieldTerrain&&) = delete;

		~STPHeightfieldTerrain() = default;

		void setMesh(const STPEnvironment::STPMeshSetting&);

	};

}
#endif//_STP_HEIGHTFIELD_TERRAIN_H_