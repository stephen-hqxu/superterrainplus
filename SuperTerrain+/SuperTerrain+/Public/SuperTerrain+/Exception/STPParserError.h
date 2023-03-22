#pragma once
#ifndef _STP_PARSER_ERROR_H_
#define _STP_PARSER_ERROR_H_

#include "STPFundamentalException.h"

/**
 * @param desc A more detailed description of this error.
 * @param parser_name The name of the parser.
 * @param error_title The main title of the error, should be concise.
*/
#define STP_PARSER_ERROR_CTOR_DECL const std::string&, const char*, const char*, STP_EXCEPTION_SOURCE_INFO_DECL

//create a syntax error during parsing
#define STP_PARSER_INVALID_SYNTAX_CREATE(DESC, PAR_NAME, TITLE, SRC_INFO) \
STP_STANDARD_EXCEPTION_CREATE(STPParserError::STPInvalidSyntax, SRC_INFO, DESC, PAR_NAME, TITLE)
//create a semantic error during parsing
#define STP_PARSER_SEMANTIC_ERROR_CREATE(DESC, PAR_NAME, TITLE) \
STP_STANDARD_EXCEPTION_CREATE(STPParserError::STPSemanticError, DESC, PAR_NAME, TITLE)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPParserError signals an error during parsing of script source file.
	*/
	namespace STPParserError {

		/**
		 * @brief The base of all parser error.
		*/
		class STP_API STPBasic : public STPFundamentalException::STPBasic {
		public:

			//what parser is this?
			const std::string ParserName;
			//a descriptive title of the parser error
			const std::string ErrorTitle;

			STPBasic(STP_PARSER_ERROR_CTOR_DECL);

			~STPBasic() = default;

		};

		/**
		 * @brief STPInvalidSyntax indicates an erroneous syntax in the script.
		*/
		class STP_API STPInvalidSyntax : public STPBasic {
		public:

			/**
			 * @brief Information regarding the parsing source script, (not about the cpp source code, this is captured in the base class).
			*/
			const struct STPSourceInformation {
			public:

				std::string SourceName;
				size_t Line, Column;

			} SourceInformation;

			/**
			 * @param source_info The information of the parsing source.
			*/
			STPInvalidSyntax(const STPSourceInformation&, STP_PARSER_ERROR_CTOR_DECL);

			~STPInvalidSyntax() = default;

		};

		/**
		 * @brief STPSemanticError indicates an erroneous semantic that prevents the parser from putting everything together.
		*/
		class STP_API STPSemanticError : public STPBasic {
		public:

			STPSemanticError(STP_PARSER_ERROR_CTOR_DECL);

			~STPSemanticError() = default;

		};

	}

}

#undef STP_PARSER_ERROR_CTOR_DECL
#endif//_STP_PARSER_ERROR_H_