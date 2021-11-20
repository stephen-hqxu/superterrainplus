#ifndef _STP_INVALID_SYNTAX_H_
#define _STP_INVALID_SYNTAX_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPInvalidSyntax indicates there is an invalid syntax during lexing and parsing of SuperTerrain+ custom languages.
	*/
	class STP_API STPInvalidSyntax : public std::runtime_error {
	public:

		/**
		 * @brief Init STPInvalidSyntax
		 * @param msg Message about the invalid syntax
		*/
		explicit STPInvalidSyntax(const char*);

	};

}
#endif//_STP_INVALID_SYNTAX_H_