#pragma once
#ifndef _STP_INI_STRING_H_
#define _STP_INI_STRING_H_

#include <string>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIString is a string value in INI for converting string to primitive data type.
	*/
	struct STPINIString : public std::string {
	public:

		/**
		 * @brief Initialise an empty STPINIString instance.
		*/
		STPINIString() = default;

		/**
		 * @brief Initialise a STPINIString from a std::string instance.
		 * @param str The pointer to the string instance.
		 * It will be copied to the current instance.
		*/
		explicit STPINIString(const std::string&);

		STPINIString(const STPINIString&) = default;

		STPINIString(STPINIString&&) noexcept = default;

		STPINIString& operator=(const STPINIString&) = default;

		STPINIString& operator=(STPINIString&&) noexcept = default;

		~STPINIString() = default;

		/**
		 * @brief Convert the string literal to the target data type.
		 * @tparam T The target type. Only non-boolean primitive types are allowed.
		 * @return The converted type.
		*/
		template<typename T>
		T to() const;

	};

}
#include "STPINIString.inl"
#endif//_STP_INI_STRING_H_