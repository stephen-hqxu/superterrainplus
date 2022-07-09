//Do not use "pragma once" because declaration and definition resides in the same header.
#ifndef _STP_DEVICE_ERROR_HANDLER_BLUEPRINT_HPP_
#define _STP_DEVICE_ERROR_HANDLER_BLUEPRINT_HPP_

//Define to disable error output to error stream
#ifdef STP_DEVICE_ERROR_SUPPRESS_CERR
#undef STP_DEVICE_ERROR_SUPPRESS_CERR
#define STP_DEVICE_ERROR_SUPPRESS_CERR true
#else
#define STP_DEVICE_ERROR_SUPPRESS_CERR false
#endif

//Define to suffix a function qualifier to engine assert function
#ifndef STP_ENGINE_ASSERT_QUAL
#define STP_ENGINE_ASSERT_QUAL
#endif//STP_ENGINE_ASSERT_QUAL

namespace SuperTerrainPlus {
	/**
	 * @brief STPDeviceErrorHandlerBlueprint is a helper for defining an error handler for functions return a status code.
	*/
	namespace STPDeviceErrorHandlerBlueprint {
		
		/**
		 * @brief Check the error code and emit exception whenever applicable.
		 * Please do not call this function manually, and instead wrap it with a macro function.
		 * @param err_code The error code for the API.
		 * @param file The filename where the function is called.
		 * @param function The function name where the function is called.
		 * @param line The line where the function is called.
		 * @param no_msg Set to true to suppress error message output to the error stream.
		*/
		template<typename Err>
		STP_ENGINE_ASSERT_QUAL void assertEngine(Err, const char* __restrict, const char* __restrict, int, bool) noexcept(false);

	}
}

#define STP_ASSERT_ENGINE_BASIC(EC) \
	SuperTerrainPlus::STPDeviceErrorHandlerBlueprint::assertEngine( \
		EC, __FILE__, __FUNCTION__, __LINE__, STP_DEVICE_ERROR_SUPPRESS_CERR)

#endif//_STP_DEVICE_ERROR_HANDLER_BLUEPRINT_HPP_
//End of header

//Start of source
#ifdef STP_DEVICE_ERROR_HANDLER_BLUEPRINT_IMPLEMENTATION

//System
#include <iostream>
#include <sstream>
#include <typeinfo>

using std::ostringstream;
using std::cerr;
using std::endl;

//always throw an exception
template<class E>
[[noreturn]] static inline void printError(ostringstream& msg, bool no_msg) noexcept(false) {
	if (!no_msg) {
		cerr << msg.str();
	}
	//throw exception
	throw E(msg.str().c_str());
}

#define ASSERT_FUNCTION(ERR) template<> \
	STP_ENGINE_ASSERT_QUAL void SuperTerrainPlus::STPDeviceErrorHandlerBlueprint::assertEngine<ERR>(ERR err_code, \
		const char* __restrict file, const char* __restrict function, int line, bool no_msg) noexcept(false)
#define WRITE_ERR_STRING(SS) SS << file << "(" << function << "):" << line

#endif//STP_DEVICE_ERROR_HANDLER_BLUEPRINT_IMPLEMENTATION