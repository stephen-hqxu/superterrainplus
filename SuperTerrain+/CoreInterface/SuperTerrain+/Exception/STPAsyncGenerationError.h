#pragma once
#ifndef _STP_ASYNC_GENERATION_ERROR_H_
#define _STP_ASYNC_GENERATION_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPAsyncGenerationError indicates some exception is thrown during async launch to compute the terrain.
	*/
	class STP_API STPAsyncGenerationError : public std::runtime_error {
	public:

		/**
		 * @brief Init STPAsyncGenerationError
		 * @param msg The mssage about the async failure
		*/
		explicit STPAsyncGenerationError(const char*);

	};

}
#endif//_STP_ASYNC_GENERATION_ERROR_H_