//INLINE DEFINITION FOR INI STRING UTILITY, DO NOT INCLUDE MANUALLY
#ifdef _STP_INI_STRING_H_

#include <charconv>
#include <stdexcept>

SuperTerrainPlus::STPAlgorithm::STPINIString::STPINIString(const std::string& str) : std::string(str) {

}

template<typename T>
inline T SuperTerrainPlus::STPAlgorithm::STPINIString::to() const {
	T value = static_cast<T>(0);
	const auto [ptr, ec] = std::from_chars(this->cbegin(), this->cend(), value);

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

#endif//_STP_INI_STRING_H_