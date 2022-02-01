#pragma once
#ifndef _STP_SCREEN_H_
#define _STP_SCREEN_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../../Object/STPBuffer.h"
#include "../../Object/STPVertexArray.h"
#include "../../Object/STPShaderManager.h"

#include "../../Utility/STPLogStorage.hpp"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPScreen is a special rendering component that draws a quad at normalised device coordinate.
	 * This can be used as a helper class for off-screen renderers.
	*/
	class STP_REALISM_API STPScreen {
	public:

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

		STPScreen(STPScreen&&) = delete;

		STPScreen& operator=(const STPScreen&) = delete;

		STPScreen& operator=(STPScreen&&) = delete;

		virtual ~STPScreen() = default;

	};

}
#endif//_STP_SCREEN_H_