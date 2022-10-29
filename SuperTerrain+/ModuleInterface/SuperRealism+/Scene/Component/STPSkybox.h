#pragma once
#ifndef _STP_SKYBOX_H_
#define _STP_SKYBOX_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL
#include "../../Object/STPShaderManager.h"
#include "../../Object/STPProgramManager.h"
#include "../../Object/STPBuffer.h"
#include "../../Object/STPVertexArray.h"

#include <memory>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSkybox is a special rendering component that draws an axis-aligned cube centred at the origin.
	 * This can be used as a helper renderer for environment-based rendering.
	*/
	class STP_REALISM_API STPSkybox {
	public:

		/**
		 * @brief STPSkyboxVertexShader provides a vertex shader for skybox rendering.
		 * This vertex shader can be shared by many environment renderers during initialisation.
		 */
		class STP_REALISM_API STPSkyboxVertexShader {
		private:

			STPShaderManager SkyboxVertexShader;

		public:

			/**
			 * @brief Initialise a new skybox vertex shader.
			*/
			STPSkyboxVertexShader();

			STPSkyboxVertexShader(const STPSkyboxVertexShader&) = delete;

			STPSkyboxVertexShader(STPSkyboxVertexShader&&) noexcept = default;

			STPSkyboxVertexShader& operator=(const STPSkyboxVertexShader&) = delete;

			STPSkyboxVertexShader& operator=(STPSkyboxVertexShader&&) noexcept = default;

			~STPSkyboxVertexShader() = default;

			/**
			 * @brief Get the underlying skybox shader manager.
			 * @return The pointer to the skybox vertex shader.
			*/
			const STPShaderManager& operator*() const;

		};

		/**
		 * @brief STPSkyboxVertexBuffer provides vertex buffer of a skybox for drawing.
		*/
		class STP_REALISM_API STPSkyboxVertexBuffer {
		private:

			//A skybox buffer contains all vertices and UV coordinate of a cube.
			STPBuffer SkyboxBuffer, SkyboxIndex, SkyboxDrawCommand;
			STPVertexArray SkyboxArray;

		public:

			/**
			 * @brief Initialise a new skybox vertex buffer.
			*/
			STPSkyboxVertexBuffer();

			STPSkyboxVertexBuffer(const STPSkyboxVertexBuffer&) = delete;

			STPSkyboxVertexBuffer(STPSkyboxVertexBuffer&&) noexcept = default;

			STPSkyboxVertexBuffer& operator=(const STPSkyboxVertexBuffer&) = delete;

			STPSkyboxVertexBuffer& operator=(STPSkyboxVertexBuffer&&) noexcept = default;

			~STPSkyboxVertexBuffer() = default;

			/**
			 * @brief Bind the skybox vertex buffer.
			*/
			void bind() const;

		};

		/**
		 * @brief STPSkyboxInitialiser contains information required to create a skybox-based renderer.
		*/
		struct STPSkyboxInitialiser {
		public:

			//Specifies a pointer to a skybox vertex shader.
			//This shader can be shared by many skybox renderer during initialisation and no state is retained afterwards,
			//therefore it can be destroyed safely.
			const STPSkyboxVertexShader* VertexShader;
			//The skybox vertex buffer to be shared between different skybox renderers.
			std::weak_ptr<const STPSkyboxVertexBuffer> SharedVertexBuffer;

		};

	private:

		std::shared_ptr<const STPSkyboxVertexBuffer> SkyboxVertex;

	public:

		mutable STPProgramManager SkyboxRenderer;

		/**
		 * @brief Initialise a skybox renderer instance.
		*/
		STPSkybox() = default;

		STPSkybox(const STPSkybox&) = delete;

		STPSkybox(STPSkybox&&) noexcept = default;

		STPSkybox& operator=(const STPSkybox&) = delete;

		STPSkybox& operator=(STPSkybox&&) noexcept = default;

		~STPSkybox() = default;

		/**
		 * @brief Initialise the skybox renderer.
		 * All previous initialisation states are abandoned.
		 * @param skybox_fs The fragment shader for the skybox rendering.
		 * @param skybox_init The pointer to the skybox initialiser.
		*/
		void initSkyboxRenderer(const STPShaderManager&, const STPSkyboxInitialiser&);

		/**
		 * @brief Draw the skybox using the skybox rendering program.
		 * All buffers and program are bound automatically.
		*/
		void drawSkybox() const;

	};
}
#endif//_STP_SKYBOX_H_