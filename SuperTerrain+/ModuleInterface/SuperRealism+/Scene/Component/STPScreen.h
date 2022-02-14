#pragma once
#ifndef _STP_SCREEN_H_
#define _STP_SCREEN_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../../Object/STPBuffer.h"
#include "../../Object/STPVertexArray.h"
#include "../../Object/STPFrameBuffer.h"
#include "../../Object/STPTexture.h"

#include "../../Object/STPShaderManager.h"
#include "../../Object/STPProgramManager.h"

#include "../../Utility/STPLogStorage.hpp"

//GLM
#include <glm/vec2.hpp>

//System
#include <memory>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPScreen is a special rendering component that draws a quad at normalised device coordinate.
	 * This can be used as a helper class for off-screen renderers.
	*/
	class STP_REALISM_API STPScreen {
	public:

		/**
		 * @brief STPSimpleScreenFrameBuffer is a simple wrapper for off-screen rendering.
		 * It allows quick setup and render to a single colour-attached framebuffer with stencil buffer for stencil testing.
		*/
		class STP_REALISM_API STPSimpleScreenFrameBuffer {
		public:

			//Off-screen rendering buffer
			STPTexture ScreenColor;
			STPFrameBuffer ScreenColorContainer;

			/**
			 * @brief Initialise a new simple screen framebuffer instance.
			*/
			STPSimpleScreenFrameBuffer();

			STPSimpleScreenFrameBuffer(const STPSimpleScreenFrameBuffer&) = delete;

			STPSimpleScreenFrameBuffer(STPSimpleScreenFrameBuffer&&) noexcept = default;

			STPSimpleScreenFrameBuffer& operator=(const STPSimpleScreenFrameBuffer&) = delete;

			STPSimpleScreenFrameBuffer& operator=(STPSimpleScreenFrameBuffer&&) noexcept = default;

			~STPSimpleScreenFrameBuffer() = default;

			/**
			 * @brief Setup the screen buffer property.
			 * This function triggers reallocation to the internal buffer, therefore it is considered to be expensive.
			 * @param stencil The pointer to the stencil buffer to be attached to the internal framebuffer.
			 * This allows specifying which region to perform occlusion.
			 * A nullptr means no stencil buffer is provided and occlusion will be performed to the entire screen.
			 * @param dimension The new dimension for the rendering screen.
			 * @param internal Specifies the sized internal format for the new screen buffer.
			*/
			void setScreenBuffer(STPTexture*, const glm::uvec2&, STPOpenGL::STPenum);

			/**
			 * @brief Clear the color attachment of the screen buffer.
			 * @param color The color to be cleared to.
			*/
			void clearScreenBuffer(const glm::vec4&);

			/**
			 * @brief Activating the screen color framebuffer.
			 * To deactivate, bind framebuffer target to any other points.
			*/
			void capture() const;

		};

		/**
		 * @brief STPScreenVertexShader provides a vertex shader for off-screen rendering.
		 * This vertex shader can be shared by many screen renderers during initialisation.
		*/
		class STP_REALISM_API STPScreenVertexShader {
		private:

			STPShaderManager ScreenVertexShader;

		public:

			typedef STPLogStorage<1ull> STPScreenVertexShaderLog;

			/**
			 * @brief Init and compile a screen vertex shader.
			 * @param log The pointer to the log to store shader compilation result.
			*/
			STPScreenVertexShader(STPScreenVertexShaderLog&);

			STPScreenVertexShader(const STPScreenVertexShader&) = delete;

			STPScreenVertexShader(STPScreenVertexShader&&) noexcept = default;

			STPScreenVertexShader& operator=(const STPScreenVertexShader&) = delete;

			STPScreenVertexShader& operator=(STPScreenVertexShader&&) noexcept = default;

			~STPScreenVertexShader() = default;

			/**
			 * @brief Get the underlying screen vertex shader.
			 * @return The pointer to the vertex shader that is ready to be linked with a complete pipeline.
			*/
			const STPShaderManager& operator*() const;

		};

		/**
		 * @brief STPScreenVertexBuffer provides vertex buffer objects for screen drawing.
		 * It is recommended to share this buffer among different screen instances within the same context.
		 * The vertex buffer is not allowed to be moved to avoid making all referenced screen instance holding a deleted reference.
		*/
		class STP_REALISM_API STPScreenVertexBuffer {
		private:

			//The buffer to represent the off-screen rendering screen
			STPBuffer ScreenBuffer, ScreenIndex, ScreenRenderCommand;
			STPVertexArray ScreenArray;

		public:

			/**
			 * @brief Init a new screen vertex buffer.
			*/
			STPScreenVertexBuffer();

			STPScreenVertexBuffer(const STPScreenVertexBuffer&) = delete;

			STPScreenVertexBuffer(STPScreenVertexBuffer&&) = delete;

			STPScreenVertexBuffer& operator=(const STPScreenVertexBuffer&) = delete;

			STPScreenVertexBuffer& operator=(STPScreenVertexBuffer&&) = delete;

			~STPScreenVertexBuffer() = default;

			/**
			 * @brief Bind the screen vertex buffer.
			*/
			void bind() const;

		};
		typedef std::shared_ptr<const STPScreenVertexBuffer> STPSharableScreenVertexBuffer;

		typedef STPLogStorage<2ull> STPScreenLog;

		/**
		 * @brief Information necessary to create any screen instance.
		 * None of the underlying pointer should be null.
		*/
		struct STPScreenInitialiser {
		public:

			//The vertex shader to be shared during initialisation.
			//This vertex shader can be safely destroyed after initialisation and no reference is retained.
			const STPScreenVertexShader* VertexShader;
			//The vertex buffer to be shared with different screen instances.
			//This buffer is managed by all shared instances automatically and the caller
			//do not have to retain the buffer.
			const STPSharableScreenVertexBuffer* SharedVertexBuffer;
			//The output log from shader compilation during initialisation.
			STPScreenLog* Log;

		};

	protected:

		const STPSharableScreenVertexBuffer ScreenVertex;

		STPProgramManager OffScreenRenderer;

		/**
		 * @brief Initialise the off-screen renderer.
		 * All old states in the previous screen renderer, if any, is lost and the program is recompiled.
		 * @param screen_fs_source The pointer to the fragment shader source code to be compiled and used by the pipeline.
		 * @param screen_init The pointer to the screen initialiser.
		*/
		void initScreenRenderer(const STPShaderManager::STPShaderSource&, const STPScreenInitialiser&);

		/**
		 * @brief Draw the screen.
		 * Buffer and program is not bound and used automatically.
		*/
		void drawScreen() const;

	public:

		/**
		 * @brief Initialise a screen renderer helper instance.
		 * @param screen_vb The pointer to screen vertex buffer to be shared with this screen class.
		*/
		STPScreen(const STPSharableScreenVertexBuffer&);

		STPScreen(const STPScreen&) = delete;

		STPScreen(STPScreen&&) noexcept = default;

		STPScreen& operator=(const STPScreen&) = delete;

		STPScreen& operator=(STPScreen&&) noexcept = default;

		virtual ~STPScreen() = default;

	};

}
#endif//_STP_SCREEN_H_