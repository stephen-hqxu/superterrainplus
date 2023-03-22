#pragma once
#ifndef _STP_CUDA_ERROR_H_
#define _STP_CUDA_ERROR_H_

#include "../STPFundamentalException.h"

//create a manual CUDA error with custom error message
#define STP_CUDA_ERROR_CREATE(MSG) STP_STANDARD_EXCEPTION_CREATE(STPCUDAError, MSG)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPCUDAError signals an error during execution of CUDA API.
	*/
	class STP_API STPCUDAError : public STPFundamentalException::STPBasic {
	public:

		/**
		 * @param err_str The error string from CUDA API.
		*/
		STPCUDAError(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPCUDAError() = default;

	};

}
#endif//_STP_CUDA_ERROR_H_