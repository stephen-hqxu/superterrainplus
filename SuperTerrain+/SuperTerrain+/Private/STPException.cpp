#include <SuperTerrain+/Exception/API/STPCUDAError.h>
#include <SuperTerrain+/Exception/API/STPGLError.h>
#include <SuperTerrain+/Exception/API/STPSQLError.h>

#include <SuperTerrain+/Exception/STPFundamentalException.h>
#include <SuperTerrain+/Exception/STPInsufficientMemory.h>
#include <SuperTerrain+/Exception/STPInvalidEnum.h>
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPIOException.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
#include <SuperTerrain+/Exception/STPParserError.h>
#include <SuperTerrain+/Exception/STPUnimplementedFeature.h>
#include <SuperTerrain+/Exception/STPUnsupportedSystem.h>
#include <SuperTerrain+/Exception/STPValidationFailed.h>

#include <cstdint>
//String
#include <iomanip>
#include <sstream>

using std::string;
using std::ostringstream;

using std::to_string;
using std::endl;

using namespace std::string_literals;

using namespace SuperTerrainPlus::STPException;

constexpr static std::streamsize SeparatorWidth = 80;

#define STP_EXCEPTION_SOURCE_INFO_DEF const char* const source, const char* const function, const int line
#define STP_EXCEPTION_SOURCE_INFO_ARG source, function, line

/* STPCUDAError.h */

STPCUDAError::STPCUDAError(const string& err_str, STP_EXCEPTION_SOURCE_INFO_DEF) : STPBasic(err_str, STP_EXCEPTION_SOURCE_INFO_ARG) {

}

/* STPGLError.h */

STPGLError::STPGLError(const string& desc, STP_EXCEPTION_SOURCE_INFO_DEF) : STPBasic(desc, STP_EXCEPTION_SOURCE_INFO_ARG) {

}

/* STPSQLError.h */

STPSQLError::STPSQLError(const string& err_str, STP_EXCEPTION_SOURCE_INFO_DEF) : STPBasic(err_str, STP_EXCEPTION_SOURCE_INFO_ARG) {

}

/* STPFundamentalException.h */

STPFundamentalException::STPBasic::STPBasic(const string& description, STP_EXCEPTION_SOURCE_INFO_DEF) :
	Description(description), SourceFilename(source), FunctionName(function), Line(line) {
	ostringstream msg;
	msg << this->SourceFilename << '(' << this->FunctionName << "):" << this->Line << endl;
	//add a fancy horizontal line
	msg << std::setfill('-') << std::setw(SeparatorWidth) << '\n';
	msg << this->Description << endl;

	this->Message = msg.str();
}

const char* STPFundamentalException::STPBasic::what() const noexcept {
	return this->Message.c_str();
}

STPFundamentalException::STPAssertion::STPAssertion(const char* const expression, const string& description,
	STP_EXCEPTION_SOURCE_INFO_DEF) : STPBasic(description, STP_EXCEPTION_SOURCE_INFO_ARG) {
	ostringstream msg;
	msg << "Assertion Failed:\n" << expression << endl;
	//horizontal bar
	msg << std::setfill('.') << std::setw(SeparatorWidth) << '\n';
	msg.flush();

	//add to the original message
	this->Message.insert(0u, msg.str());
}

/* STPInsufficientMemory.h */

inline static string createInsufficientMemoryMessage(const size_t current, const size_t request, const size_t max, const char* const unit) {
	ostringstream msg;
	msg << "Based on current memory usage of "s << current
		<< ", requested "s << request << " amount of memory exceeds the maximum allowance of "s << max
		<< " for the current system; memory unit in \'"s << unit << "\'"s << endl;
	return msg.str();
}

STPInsufficientMemory::STPInsufficientMemory(const char* const expr, const size_t current_memory, const size_t request_memory,
	const size_t max_memory, const char* const unit, STP_EXCEPTION_SOURCE_INFO_DEF) :
	//lifetime of string will be extended
	STPAssertion(expr, createInsufficientMemoryMessage(current_memory, request_memory, max_memory, unit), STP_EXCEPTION_SOURCE_INFO_ARG),
	CurrentMemory(current_memory), RequestMemory(request_memory), MaxMemory(max_memory), MemoryUnit(unit) {

}

/* STPInvalidEnum.h */

STPInvalidEnum::STPInvalidEnum(const string& enum_value, const char* const enum_class, STP_EXCEPTION_SOURCE_INFO_DEF) :
	STPBasic("Enum value \'"s + enum_value + "\' is not a defined valid enum in enum definition \'"s + enum_class + "\'"s, STP_EXCEPTION_SOURCE_INFO_ARG),
	Value(enum_value), Class(enum_class) {

}

/* STPInvalidEnvironment.h */

STPInvalidEnvironment::STPInvalidEnvironment(const char* const expression, const char* const env_name, STP_EXCEPTION_SOURCE_INFO_DEF) :
	STPAssertion(expression, "Environment \'"s + string(env_name) + "\' validation fails"s, STP_EXCEPTION_SOURCE_INFO_ARG), Environment(env_name) {

}

/* STPIOException.h */

STPIOException::STPIOException(const string& msg, STP_EXCEPTION_SOURCE_INFO_DEF) : STPBasic(msg, STP_EXCEPTION_SOURCE_INFO_ARG) {

}

/* STPNumericDomainError.h */

STPNumericDomainError::STPNumericDomainError(const char* const expression, const string& explaination, STP_EXCEPTION_SOURCE_INFO_DEF) :
	STPAssertion(expression, explaination, STP_EXCEPTION_SOURCE_INFO_ARG) {

}

/* STPParserError.h */

inline static string createParserInvalidSyntaxMessage(const STPParserError::STPInvalidSyntax::STPSourceInformation& info, const std::string& desc) {
	const auto& [src_name, line, col] = info;
	ostringstream msg;

	msg << src_name << ":(Line: " << line << ", Column: " << col << "):" << endl;
	msg << desc << endl;

	return msg.str();
}

#define STP_PARSER_ERROR_CTOR_DEF const string& desc, const char* const parser_name, const char* const error_title, STP_EXCEPTION_SOURCE_INFO_DEF
#define STP_PARSER_ERROR_CTOR_ARG parser_name, error_title, STP_EXCEPTION_SOURCE_INFO_ARG

STPParserError::STPBasic::STPBasic(STP_PARSER_ERROR_CTOR_DEF) :
	STPFundamentalException::STPBasic(string(parser_name) + ":<"s + string(error_title) + ">:\n"s + desc, STP_EXCEPTION_SOURCE_INFO_ARG),
	ParserName(parser_name), ErrorTitle(error_title) {

}

STPParserError::STPInvalidSyntax::STPInvalidSyntax(const STPSourceInformation& source_info, STP_PARSER_ERROR_CTOR_DEF) :
	STPBasic(createParserInvalidSyntaxMessage(source_info, desc), STP_PARSER_ERROR_CTOR_ARG), SourceInformation(source_info) {

}

STPParserError::STPSemanticError::STPSemanticError(STP_PARSER_ERROR_CTOR_DEF) : STPBasic(desc, STP_PARSER_ERROR_CTOR_ARG) {

}

/* STPUnimplementedFeature.h */

STPUnimplementedFeature::STPUnimplementedFeature(const string& feature_desc, STP_EXCEPTION_SOURCE_INFO_DEF) :
	STPBasic("The following feature is not yet implemented:\n"s + feature_desc, STP_EXCEPTION_SOURCE_INFO_ARG), Feature(feature_desc) {

}

/* STPUnsupportedSystem.h */

STPUnsupportedSystem::STPUnsupportedSystem(const string& violated_req, STP_EXCEPTION_SOURCE_INFO_DEF) :
	STPBasic("The following minimum system requirement is not satisfied on this platform:\n"s + violated_req, STP_EXCEPTION_SOURCE_INFO_ARG),
	ViolatedRequirement(violated_req) {

}

/* STPValidationFailed.h */

STPValidationFailed::STPValidationFailed(const char* const expr, const string& msg, STP_EXCEPTION_SOURCE_INFO_DEF) :
	STPAssertion(expr, msg, STP_EXCEPTION_SOURCE_INFO_ARG) {

}