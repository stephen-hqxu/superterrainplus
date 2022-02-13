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

#include "../../Utility/STPLogStorage.hpp"

//GLM
#include <glm/vec2.hpp>

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

		typedef STPLogStorage<1ull> STPScreenLog;

	private:

		//The buffer to represent the off-screen rendering screen
		STPBuffer ScreenBuffer, ScreenIndex, ScreenRenderCommand;
		STPVertexArray ScreenArray;

	protected:

		/**
		 * @brief Compile and return a vertex shader for screen rendering.
		 * @param log The pointer to the log to store shader compilation result.
		 * @return A vertex shader for screen rendering. 
		*/
		static STPShaderManager compileScreenVertexShader(STPScreenLog& log);

		/**
		 * @brief Draw the screen.
		 * Vertex buffer is bounded automatically.
		*/
		void drawScreen() const;

	public:

		/**
		 * @brief Initialise a screen renderer helper instance.
		*/
		STPScreen();

		STPScreen(const STPScreen&) = delete;

		STPScreen(STPScreen&&) noexcept = default;

		STPScreen& operator=(const STPScreen&) = delete;

		STPScreen& operator=(STPScreen&&) noexcept = default;

		virtual ~STPScreen() = default;

	};

}
#endif//_STP_SCREEN_H_