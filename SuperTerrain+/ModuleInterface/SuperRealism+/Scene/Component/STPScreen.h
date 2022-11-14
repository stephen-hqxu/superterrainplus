#pragma once
#ifndef _STP_SCREEN_H_
#define _STP_SCREEN_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../../Object/STPBuffer.h"
#include "../../Object/STPVertexArray.h"
#include "../../Object/STPFrameBuffer.h"
#include "../../Object/STPSampler.h"
#include "../../Object/STPBindlessTexture.h"

#include "../../Object/STPShaderManager.h"
#include "../../Object/STPProgramManager.h"

//GLM
#include <glm/vec2.hpp>

//System
#include <memory>
#include <functional>

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

		protected:

			/**
			 * @brief Reattach a newly allocated texture to the framebuffer.
			 * @param stencil The stencil buffer to be bound.
			 * @param dimension The size of the new texture.
			 * @param internal The internal format of the new texture.
			 * @return The newly allocated texture which has been attached.
			*/
			STPTexture updateScreenFrameBuffer(STPTexture*, const glm::uvec2&, STPOpenGL::STPenum);

		public:

			/**
			 * @brief Initialise a new simple screen framebuffer instance.
			*/
			STPSimpleScreenFrameBuffer() noexcept;

			STPSimpleScreenFrameBuffer(const STPSimpleScreenFrameBuffer&) = delete;

			STPSimpleScreenFrameBuffer(STPSimpleScreenFrameBuffer&&) noexcept = default;

			STPSimpleScreenFrameBuffer& operator=(const STPSimpleScreenFrameBuffer&) = delete;

			STPSimpleScreenFrameBuffer& operator=(STPSimpleScreenFrameBuffer&&) noexcept = default;

			virtual ~STPSimpleScreenFrameBuffer() = default;

			/**
			 * @brief Setup the screen buffer property.
			 * This function triggers reallocation to the internal buffer, therefore it is considered to be expensive.
			 * @param stencil The pointer to the stencil buffer to be attached to the internal framebuffer.
			 * This allows specifying which region to perform occlusion.
			 * A nullptr means no stencil buffer is provided and occlusion will be performed to the entire screen.
			 * @param dimension The new dimension for the rendering screen.
			 * @param internal Specifies the sized internal format for the new screen buffer.
			*/
			virtual void setScreenBuffer(STPTexture*, const glm::uvec2&, STPOpenGL::STPenum);

			/**
			 * @brief Clear the colour attachment of the screen buffer.
			 * @param colour The colour to be cleared to.
			*/
			void clearScreenBuffer(const glm::vec4&) noexcept;

			/**
			 * @brief Activating the screen colour framebuffer.
			 * To deactivate, bind framebuffer target to any other points.
			*/
			void capture() const noexcept;

		};

		/**
		 * @brief Similar to STPSimpleScreenFrameBuffer, except the texture contains a bindless handle.
		 * @see STPSimpleScreenFrameBuffer
		*/
		class STP_REALISM_API STPSimpleScreenBindlessFrameBuffer : public STPSimpleScreenFrameBuffer {
		public:

			//The bindless handle for the screen colour texture.
			STPSampler ScreenColorSampler;
			STPBindlessTexture::STPHandle ScreenColorHandle;

			STPSimpleScreenBindlessFrameBuffer() noexcept;

			STPSimpleScreenBindlessFrameBuffer(const STPSimpleScreenBindlessFrameBuffer&) = delete;

			STPSimpleScreenBindlessFrameBuffer(STPSimpleScreenBindlessFrameBuffer&&) noexcept = default;

			STPSimpleScreenBindlessFrameBuffer& operator=(const STPSimpleScreenBindlessFrameBuffer&) = delete;

			STPSimpleScreenBindlessFrameBuffer& operator=(STPSimpleScreenBindlessFrameBuffer&&) noexcept = default;

			~STPSimpleScreenBindlessFrameBuffer() = default;

			//@see STPSimpleScreenFrameBuffer::setScreenBuffer
			void setScreenBuffer(STPTexture*, const glm::uvec2&, STPOpenGL::STPenum) override;

		};

		/**
		 * @brief STPScreenVertexBuffer provides vertex buffer objects for screen drawing.
		*/
		class STP_REALISM_API STPScreenVertexBuffer {
		private:

			//The buffer to represent the off-screen rendering screen
			STPBuffer ScreenBuffer, ScreenRenderCommand;
			STPVertexArray ScreenArray;

		public:

			/**
			 * @brief Init a new screen vertex buffer.
			*/
			STPScreenVertexBuffer() noexcept;

			STPScreenVertexBuffer(const STPScreenVertexBuffer&) = delete;

			STPScreenVertexBuffer(STPScreenVertexBuffer&&) noexcept = default;

			STPScreenVertexBuffer& operator=(const STPScreenVertexBuffer&) = delete;

			STPScreenVertexBuffer& operator=(STPScreenVertexBuffer&&) noexcept = default;

			~STPScreenVertexBuffer() = default;

			/**
			 * @brief Bind the screen vertex buffer.
			*/
			void bind() const noexcept;

		};

		/**
		 * @brief Information necessary to create any screen instance.
		 * None of the underlying pointer should be null.
		*/
		struct STP_REALISM_API STPScreenInitialiser {
		public:

			//The vertex shader to be shared during initialisation.
			//This vertex shader can be safely destroyed after initialisation and no reference is retained.
			const STPShaderManager::STPShader VertexShader;
			//The vertex buffer to be shared with different screen instances.
			std::weak_ptr<const STPScreenVertexBuffer> SharedVertexBuffer;

			/**
			 * @brief Initialise and compile a screen vertex shader.
			*/
			STPScreenInitialiser();

			~STPScreenInitialiser() = default;

		};

	private:

		std::shared_ptr<const STPScreenVertexBuffer> ScreenVertex;

	public:

		/**
		 * @brief STPScreenProgramExecutor is a smart guard over a function that uses the screen drawing program.
		 * It allows the function to issue multiple draw call from the same program without re-using the program repetitively.
		 * At the end of execution states are cleared up automatically to avoid state leakage.
		*/
		struct STPScreenProgramExecutor {
		public:

			//This is the thing that prevents state leakage for our screen renderer.
			const STPProgramManager::STPProgramStateManager OffScreenRendererState;

			/**
			 * @brief Start a screen program rendering execution.
			 * The screen draw program is made active automatically and will be deactivated when the current instance is destroyed.
			 * Changing any indirect buffer, vertex array and program state while a program executor is active will lead to undefined behaviour.
			 * @param screen The screen program to be rendered.
			*/
			STPScreenProgramExecutor(const STPScreen&) noexcept;

			STPScreenProgramExecutor(const STPScreenProgramExecutor&) = delete;

			STPScreenProgramExecutor(STPScreenProgramExecutor&&) = delete;

			STPScreenProgramExecutor& operator=(const STPScreenProgramExecutor&) = delete;

			STPScreenProgramExecutor& operator=(STPScreenProgramExecutor&&) = delete;

			~STPScreenProgramExecutor() = default;

			/**
			 * @brief Draw the screen.
			*/
			void operator()() const noexcept;

		};

		STPProgramManager OffScreenRenderer;

		/**
		 * @brief Initialise a screen renderer helper instance.
		*/
		STPScreen() = default;

		STPScreen(const STPScreen&) = delete;

		STPScreen(STPScreen&&) noexcept = default;

		STPScreen& operator=(const STPScreen&) = delete;

		STPScreen& operator=(STPScreen&&) noexcept = default;

		~STPScreen() = default;

		/**
		 * @brief Initialise the off-screen renderer.
		 * All old states in the previous screen renderer, if any, is lost and the program is recompiled.
		 * It is a undefined behaviour if any member variables are used before this function is called for the first time
		 * since object initialisation.
		 * @param screen_fs The pointer to the fragment shader used by the pipeline.
		 * @param screen_init The pointer to the screen initialiser.
		*/
		void initScreenRenderer(const STPShaderManager::STPShader&, const STPScreenInitialiser&);

		/**
		 * @brief Draw the screen.
		 * Buffer and program are bound and used automatically.
		*/
		void drawScreen() const noexcept;

		/**
		 * @brief Draw the screen using a screen draw executor.
		 * Buffer and program states are bound upon the executor is returned, and cleared up automatically at the end.
		 * This allows issuing multiple draw commands without reactivating the states repetitively.
		 * @return A screen draw executor.
		*/
		[[nodiscard]] STPScreenProgramExecutor drawScreenFromExecutor() const noexcept;

	};

}
#endif//_STP_SCREEN_H_