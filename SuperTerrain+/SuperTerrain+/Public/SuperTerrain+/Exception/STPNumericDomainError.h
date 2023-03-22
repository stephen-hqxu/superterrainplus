#pragma once
#ifndef _STP_NUMERIC_DOMAIN_ERROR_H_
#define _STP_NUMERIC_DOMAIN_ERROR_H_

#include "STPFundamentalException.h"

//check the validity of a numeric domain, provided a message if validation fails
#define STP_ASSERTION_NUMERIC_DOMAIN(EXPR, MSG) STP_ASSERTION_EXCEPTION(STPNumericDomainError, EXPR, MSG)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPNumericDomainError indicates a numeric value provided is out of valid range.
	*/
	class STP_API STPNumericDomainError : public STPFundamentalException::STPAssertion {
	public:

		/**
		 * @param expression
		 * @param explanation Some information provided to user about what's the correct numeric domain.
		*/
		STPNumericDomainError(const char*, const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPNumericDomainError() = default;

	};

}
#endif//_STP_NUMERIC_DOMAIN_ERROR_H_