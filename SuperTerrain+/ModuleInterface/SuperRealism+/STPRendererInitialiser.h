#pragma once
#ifndef _STP_RENDERER_INITIALISER_H_
#define _STP_RENDERER_INITIALISER_H_

#include <SuperRealism+/STPRealismDefine.h>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPRendererInitialiser is a simple initialiser that intialise all rendering components in the realism engine.
	*/
	namespace STPRendererInitialiser {

		/**
		 * @brief Init the SuperRealism+ renderer.
		 * There are a few things to be noted:
		 * - It does not initialise a GL context automatically, please either initialise it manually before calling this function, 
		 * or by initialising using SuperTerrain+ initialiser.
		 * - It only initialises the renderer on the current context.
		*/
		STP_REALISM_API void init();

	}

}
#endif//_STP_RENDERER_INITIALISER_H_