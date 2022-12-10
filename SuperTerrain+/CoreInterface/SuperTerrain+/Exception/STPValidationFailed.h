#pragma once
#ifndef _STP_VALIDATION_FAILED_H_
#define _STP_VALIDATION_FAILED_H_

#include "STPFundamentalException.h"

//assert on an expression, if failed provides an informational piece of message
#define STP_ASSERTION_VALIDATION(EXPR, MSG) STP_ASSERTION_EXCEPTION(STPValidationFailed, EXPR, MSG)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPValidationFailed is an exception for general validation error.
	*/
	class STP_API STPValidationFailed : public STPFundamentalException::STPAssertion {
	public:

		//same as the base class
		STPValidationFailed(const char*, const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPValidationFailed() = default;

	};

}
#endif//_STP_VALIDATION_FAILED_H_