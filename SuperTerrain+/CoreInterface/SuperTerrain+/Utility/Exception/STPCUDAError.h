#pragma once
#ifndef _STP_CUDA_ERROR_H_
#define _STP_CUDA_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPCUDAError specifies error thrown from CUDA API.
	*/
	class STP_API STPCUDAError : public std::runtime_error {
	public:

		/**
		 * @brief Init STPCUDAError
		 * @param msg The CUDA error message
		*/
		explicit STPCUDAError(const char*);

	};

}
#endif//_STP_CUDA_ERROR_H_