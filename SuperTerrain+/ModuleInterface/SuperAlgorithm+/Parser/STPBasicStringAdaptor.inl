//INLINE DEFINITIONS FOR STRING ADAPTOR USED BY THE PARSER
#ifdef _STP_BASIC_STRING_ADAPTOR_H_

//String Conversion
#include <charconv>

//Error
#include <SuperTerrain+/Exception/STPParserError.h>

template<class Str>
template<class... Arg, typename, bool IsNoexcept>
inline SuperTerrainPlus::STPAlgorithm::STPBasicStringAdaptor<Str>::STPBasicStringAdaptor(Arg&&... arg) noexcept(IsNoexcept) : String(std::forward<Arg>(arg)...) {

}

template<class Str>
inline Str& SuperTerrainPlus::STPAlgorithm::STPBasicStringAdaptor<Str>::operator*() noexcept {
	return this->String;
}

template<class Str>
inline const Str& SuperTerrainPlus::STPAlgorithm::STPBasicStringAdaptor<Str>::operator*() const noexcept {
	return this->String;
}

template<class Str>
inline Str* SuperTerrainPlus::STPAlgorithm::STPBasicStringAdaptor<Str>::operator->() noexcept {
	return &this->String;
}

template<class Str>
inline const Str* SuperTerrainPlus::STPAlgorithm::STPBasicStringAdaptor<Str>::operator->() const noexcept {
	return &this->String;
}

template<class Str>
template<typename T, typename UnconvertibleType>
inline T SuperTerrainPlus::STPAlgorithm::STPBasicStringAdaptor<Str>::to() const {
	constexpr static char StringLexicalConverterName[] = "STPBasicStringAdaptor Lexical Parser";
	using std::conjunction;
	using std::disjunction_v;
	using std::negation;
	using std::is_integral;
	using std::is_floating_point;
	using std::is_same;
	using std::is_same_v;

	using std::string;
	using std::string_view;

	using namespace std::string_literals;

	if constexpr (disjunction_v<conjunction<is_integral<T>, negation<is_same<T, bool>>>, is_floating_point<T>>) {
		//we want to treat boolean type separately, so exclude that
		//use standard library conversion
		T value {};
		const auto [ptr, ec] = std::from_chars(this->String.data(), this->String.data() + this->String.length(), value);
		//throw exception based on error code
		using std::errc;
		switch (ec) {
		case errc::invalid_argument:
			throw STP_PARSER_SEMANTIC_ERROR_CREATE("The target string \'"s + string(this->String)
					+ "\' does not have a numeric representation that can be converted"s,
				StringLexicalConverterName, "invalid argument");
		case errc::result_out_of_range:
			throw STP_PARSER_SEMANTIC_ERROR_CREATE(
				"The value \'"s + string(this->String)
				+ "\' is not in the range representable by the type"s, StringLexicalConverterName, "result out of range");
		default:
			//okay, no error
			return value;
		}
	} else if constexpr (is_same_v<T, bool>) {
		//string representation of boolean values
		if (this->String == "true") {
			return true;
		}
		if (this->String == "false") {
			return false;
		}

		throw STP_PARSER_SEMANTIC_ERROR_CREATE(
			"The string does not contain a valid boolean string representation", StringLexicalConverterName, "not a boolean string");
	} else if constexpr (disjunction_v<is_same<T, string>, is_same<T, string_view>>) {
		//construct a standard string object from the string adaptor content
		return T(this->String);
	}
}

#endif//_STP_BASIC_STRING_ADAPTOR_H_