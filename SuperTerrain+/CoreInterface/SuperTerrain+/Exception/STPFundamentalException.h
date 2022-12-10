#pragma once
#ifndef _STP_FUNDAMENTAL_EXCEPTION_H_
#define _STP_FUNDAMENTAL_EXCEPTION_H_

#include <SuperTerrain+/STPCoreDefine.h>

#include <exception>
#include <string>

/**
@brief Used for passing source information to the exception object.
@param source The name of the source code.
@param function The function name.
@param line The line number.
*/
#define STP_EXCEPTION_SOURCE_INFO_DECL const char*, const char*, int

//create a standard exception class with automatically encoded source information,
//this does not include description as the description may be formatted depends on different exception implementation
#define STP_STANDARD_EXCEPTION_CREATE(CLASS, ...) SuperTerrainPlus::STPException::CLASS(__VA_ARGS__, __FILE__, __FUNCTION__, __LINE__)
//create an assertion exception if the expression fails.
#define STP_ASSERTION_EXCEPTION(CLASS, EXPR, ...) \
do { \
	if (!(EXPR)) { \
		throw STP_STANDARD_EXCEPTION_CREATE(CLASS, #EXPR, __VA_ARGS__); \
	} \
} while (false)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPFundamentalException contains the basic types of all custom exceptions used by the SuperTerrain+ project.
	*/
	namespace STPFundamentalException {

		/**
		 * @brief The most basic exception type, containing source information.
		*/
		class STP_API STPBasic : public std::exception {
		public:

			//A descriptive message for the exception.
			const std::string Description;

			//The filename and function name of the source file where this exception is generated.
			const std::string SourceFilename, FunctionName;
			//The line number.
			const int Line;

			//The formatted message.
			std::string Message;

			/**
			 * @param description The descriptive message for this exception.
			*/
			STPBasic(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

			~STPBasic() = default;

			const char* what() const noexcept override;

		};

		/**
		 * @brief An assertion exception that is generated if an expression is evaluated to false. Can be used under release mode.
		*/
		class STP_API STPAssertion : public STPBasic {
		public:

			//The expression of the assertion.
			const std::string Expression;

			/**
			 * @param expression The assertion expression that fails.
			 * @param description The descriptive message for this exception.
			*/
			STPAssertion(const char*, const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

			~STPAssertion() = default;

		};

	}

}
#endif//_STP_FUNDAMENTAL_EXCEPTION_H_