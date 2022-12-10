#pragma once
#ifndef _STP_INVALID_ENUM_H_
#define _STP_INVALID_ENUM_H_

#include "STPFundamentalException.h"

#include <type_traits>

//create an invalid enum given the enum class
#define STP_INVALID_ENUM_CREATE(VALUE, CLASS) \
STP_STANDARD_EXCEPTION_CREATE(STPInvalidEnum, std::to_string(static_cast<std::underlying_type_t<CLASS>>(VALUE)), #CLASS)
//create an invalid enum given string representation of the enum
#define STP_INVALID_STRING_ENUM_CREATE(STR, ENUM_NAME) STP_STANDARD_EXCEPTION_CREATE(STPInvalidEnum, STR, ENUM_NAME)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPInvalidEnum signals if the enum value is not defined.
	*/
	class STP_API STPInvalidEnum : public STPFundamentalException::STPBasic {
	public:

		//The value of the enum and the source enum class.
		const std::string Value, Class;

		/**
		 * @param enum_value The invalid value of the enum.
		 * This value must be convertible to string.
		 * @param enum_class The enum class where the valid is undefined.
		*/
		STPInvalidEnum(const std::string&, const char*, STP_EXCEPTION_SOURCE_INFO_DECL);

	};

}
#endif//_STP_INVALID_ENUM_H_