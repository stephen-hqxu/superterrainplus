#pragma once
#ifndef _STP_INVALID_ENVIRONMENT_H_
#define _STP_INVALID_ENVIRONMENT_H_

#include "STPFundamentalException.h"

//valid the validity of an environment setting, if not throws an invalid environment exception
#define STP_ASSERTION_ENVIRONMENT(EXPR, ENV) STP_ASSERTION_EXCEPTION(STPInvalidEnvironment, EXPR, #ENV)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPInvalidEnvironment is generated if the environment setting is invalid.
	*/
	class STP_API STPInvalidEnvironment : public STPFundamentalException::STPAssertion {
	public:

		//The name of the environment where the exception is originated.
		const std::string Environment;

		/**
		 * @param expression The expression that checks the environment property which fails.
		 * @param env_name The name of the environment.
		*/
		STPInvalidEnvironment(const char*, const char*, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPInvalidEnvironment() = default;

	};

}
#endif//_STP_INVALID_ENVIRONMENT_H_