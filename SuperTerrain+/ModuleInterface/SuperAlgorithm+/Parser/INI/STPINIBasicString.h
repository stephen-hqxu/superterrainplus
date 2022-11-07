#pragma once
#ifndef _STP_INI_BASIC_STRING_H_
#define _STP_INI_BASIC_STRING_H_

#include <string>
#include <string_view>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIBasicString is a string value in INI for converting string to primitive data type.
	 * @tparam S The type of the basic string. It currently only supports std::string and std::string_view.
	*/
	template<class Str>
	struct STPINIBasicString {
	public:

		Str String;

		/**
		 * @brief Initialise an empty STPINIBasicString instance.
		*/
		STPINIBasicString() = default;

		/**
		 * @brief Initialise a STPINIBasicString from a std::string instance.
		 * @param str The pointer to the string instance.
		 * It will be copied to the current instance.
		*/
		explicit STPINIBasicString(const Str&);

		STPINIBasicString(const STPINIBasicString&) = default;

		STPINIBasicString(STPINIBasicString&&) noexcept = default;

		STPINIBasicString& operator=(const STPINIBasicString&) = default;

		STPINIBasicString& operator=(STPINIBasicString&&) noexcept = default;

		~STPINIBasicString() = default;

		/**
		 * @brief Convert the string literal to the target data type.
		 * @tparam T The target type. Only non-boolean primitive types are allowed.
		 * @return The converted type.
		*/
		template<typename T>
		T to() const;

	};

	//Owning string version of STPINIBasicString
	typedef STPINIBasicString<std::string> STPINIString;
	//Non-owning string version of STPINIBasicString
	typedef STPINIBasicString<std::string_view> STPINIStringView;

}
#include "STPINIBasicString.inl"
#endif//_STP_INI_BASIC_STRING_H_