#pragma once
#ifndef _STP_GENERIC_ERROR_HANDLER_BLUEPRINT_HPP_
#define _STP_GENERIC_ERROR_HANDLER_BLUEPRINT_HPP_

#include <sstream>

/* 
 * @brief STPGenericErrorHandlerBlueprint is a base API error handler for C API where function returns error code.
 * User needs to define a function that takes the API error code as input, and returns a string describing the error to the provided string stream.
 * This function generates an exception is the error code is not considered to be successful, as stylised by its definition.
 * @param NAME The name of the function.
 * @param ERROR_TYPE The type of the error code.
 * @param EXPECTED_VALUE Specifies a value that should be treated as "not an error" or successful.
 * @param EXCEPTION_CLASS Specifies an exception object to be generated when an error is found.
 * The exception class is constructed through a pointer to null-terminated const char and the memory should be copied.
*/
#define STP_ERROR_DESCRIPTOR(NAME, ERROR_TYPE, EXPECTED_VALUE, EXCEPTION_CLASS) \
namespace SuperTerrainPlus::STPGenericErrorHandlerBlueprint { \
	void NAME(ERROR_TYPE, std::ostringstream&); \
	inline void NAME(ERROR_TYPE error_code, const char* file, const char* function, int line) noexcept(false) { \
		if (error_code != EXPECTED_VALUE) { \
			std::ostringstream msg_str; \
			msg_str << file << "(" << function << "):" << line << '\n'; \
			NAME(error_code, msg_str); \
			msg_str << std::endl; \
			throw EXCEPTION_CLASS(msg_str.str().c_str()); \
		} \
	} \
} \
inline void SuperTerrainPlus::STPGenericErrorHandlerBlueprint::NAME(ERROR_TYPE error_code, std::ostringstream& msg_str)

/*
 * @brief This macro helps to invoke the error descriptor defined previously.
 * @param NAME The name of the error handling function to be called.
 * @param ERROR_CODE The value of the error code. This can be a literal,
 * a variable or an expression that returns the error with type defined by the error handling function.
*/
#define STP_INVOKE_ERROR_DESCRIPTOR(NAME, ERROR_CODE) \
SuperTerrainPlus::STPGenericErrorHandlerBlueprint::NAME(ERROR_CODE, __FILE__, __FUNCTION__, __LINE__)

#endif//_STP_GENERIC_ERROR_HANDLER_BLUEPRINT_HPP_