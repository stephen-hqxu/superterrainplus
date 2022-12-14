#pragma once
#ifndef _STP_BASIC_STRING_ADAPTOR_H_
#define _STP_BASIC_STRING_ADAPTOR_H_

#include <string>
#include <string_view>

#include <type_traits>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPBasicStringAdaptor is a simple adaptor for connecting between string and an arbitrary type that can be converted from string.
	 * @tparam Str The basic string class.
	*/
	template<class Str>
	class STPBasicStringAdaptor {
	private:

		//TODO: use concept and require in C++20
		//enable if the type can be parsed from a string to the desired type
		template<typename T>
		using EnableIfLexicalConvertible = std::enable_if_t<std::disjunction_v<
			//integer and bool
			std::is_integral<T>,
			//floating point
			std::is_floating_point<T>,
			//standard string objects
			std::is_same<T, std::string>,
			std::is_same<T, std::string_view>
		>>;

	public:

		//the underlying string value
		Str String;

		//default construction
		STPBasicStringAdaptor() = default;

		//in-place construction of the underlying string
		//avoid overloading matching if a copy or move constructor should be called
		template<class... Arg, typename = std::enable_if_t<std::is_constructible_v<Str, Arg...>>,
			bool IsNoexcept = std::is_nothrow_constructible_v<Str, Arg...>>
		STPBasicStringAdaptor(Arg&&...) noexcept(IsNoexcept);

		~STPBasicStringAdaptor() = default;

		/**
		 * @brief Get the pointer to the underlying string object.
		 * @return The pointer to the string object.
		*/
		Str& operator*() noexcept;

		/**
		 * @brief Get the constant pointer to the underlying string object.
		 * @return The constant pointer to the string object.
		*/
		const Str& operator*() const noexcept;

		/**
		 * @brief Dereference the underlying string object.
		 * @return The pointer to the string object.
		*/
		Str* operator->() noexcept;

		/**
		 * @brief Deference the underlying constant string object.
		 * @return The constant pointer to the string object.
		*/
		const Str* operator->() const noexcept;

		/**
		 * @brief Lexicologically parse the value of the string to a desired type.
		 * @tparam T The desired type of conversion.
		 * @tparam UnconvertibleType If you see this from your error message,
		 * it means a type that is not convertible from string is encountered.
		 * @return The converted value.
		 * An exception if generated if the value captured in the string is not parse-able to the given type.
		*/
		template<typename T, typename UnconvertibleType = EnableIfLexicalConvertible<T>>
		T to() const;

	};

	//a basic string adaptor built based on std::string
	using STPStringAdaptor = STPBasicStringAdaptor<std::string>;
	//a basic string adaptor built based on std::string_view
	using STPStringViewAdaptor = STPBasicStringAdaptor<std::string_view>;

}
#include "STPBasicStringAdaptor.inl"
#endif//_STP_BASIC_STRING_ADAPTOR_H_