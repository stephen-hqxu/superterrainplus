#pragma once
#ifndef _STP_COMPILATION_ERROR_H_
#define _STP_COMPILATION_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include "STPCUDAError.h"

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPCompilationError indicates compilation failure during runtime compilation of CUDA code.
	*/
	class STP_API STPCompilationError : public STPCUDAError {
	public:

		/**
		 * @brief Init STPCompilationError
		 * @param msg The error message from CUDA runtime compiler.
		*/
		explicit STPCompilationError(const char*);

	};

}
#endif//_STP_COMPILATION_ERROR_H_