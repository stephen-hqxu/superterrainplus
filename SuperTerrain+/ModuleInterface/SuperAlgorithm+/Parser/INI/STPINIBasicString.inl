//INLINE DEFINITION FOR INI STRING UTILITY, DO NOT INCLUDE MANUALLY
#ifdef _STP_INI_BASIC_STRING_H_

#include <charconv>
#include <stdexcept>

template<class Str>
SuperTerrainPlus::STPAlgorithm::STPINIBasicString<Str>::STPINIBasicString(const Str& str) : Str(str) {

}

template<class Str>
template<typename T>
inline T SuperTerrainPlus::STPAlgorithm::STPINIBasicString<Str>::to() const {
	T value = static_cast<T>(0);
	const auto [ptr, ec] = std::from_chars(this->data(), this->data() + this->length(), value);

	//throw exception according to the specification of std::stoi, std::stol, std::stoll
	using std::errc;
	switch (ec) {
	case errc::invalid_argument: throw std::invalid_argument("There is no pattern match with the target string");
	case errc::result_out_of_range: throw std::out_of_range("The parsed value is not in the range representable by the type");
	default:
		//no error
		return value;
	}
}

#endif//_STP_INI_BASIC_STRING_H_