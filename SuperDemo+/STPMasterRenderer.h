#pragma once
#ifndef _STP_MASTER_RENDERER_H_
#define _STP_MASTER_RENDERER_H_

//Engine Component
#include <SuperAlgorithm+/Parser/STPINIData.hpp>
#include <SuperRealism+/Utility/STPCamera.h>

#include "./Helpers/STPCommandLineOption.h"

#include <glm/vec2.hpp>

#include <memory>

namespace STPDemo {

	/**
	 * @brief Rendering the entire terrain scene for demo.
	*/
	class STPMasterRenderer {
	private:

		/**
		 * @brief STPRendererData contains data for master renderer.
		*/
		class STPRendererData;
		std::unique_ptr<STPRendererData> Data;

	public:

		/**
		 * @brief Init STPMasterRenderer.
		 * @param engine The pointer to engine INI settings.
		 * @param biome The pointer to biome INI settings.
		 * @param camera The pointer to the perspective camera for the scene.
		 * @param cmd The result from the command line parsing.
		*/
		STPMasterRenderer(const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINIStorageView&,
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINIStorageView&,
			SuperTerrainPlus::STPRealism::STPCamera&, const STPCommandLineOption::STPResult&);

		STPMasterRenderer(const STPMasterRenderer&) = delete;

		STPMasterRenderer(STPMasterRenderer&&) = delete;

		STPMasterRenderer& operator=(const STPMasterRenderer&) = delete;

		STPMasterRenderer& operator=(STPMasterRenderer&&) = delete;

		~STPMasterRenderer();

		/**
		 * @brief Main rendering functions, called every frame.
		 * @param abs_second The current frame time in second.
		 * @param delta_second The time elapsed since the last frame.
		*/
		void render(double, double);

		/**
		 * @brief Resize the post processing framebuffer.
		 * @param res The resolution of the new framebuffer.
		*/
		void resize(const glm::uvec2&);

		/**
		 * @brief Set the display gamma.
		 * @param gamma The gamma value for display.
		*/
		void setGamma(float);

	};

}
#endif//_STP_MASTER_RENDERER_H_