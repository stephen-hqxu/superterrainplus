//INLINE FUNCTION DEFINITIONS FOR SHADER MANAGER, DO NOT INCLUDE MANUALLY
#ifdef _STP_SHADER_MANAGER_H_

//System
#include <type_traits>
#include <string_view>

template<typename T>
inline SuperTerrainPlus::STPRealism::STPShaderManager::STPShaderSource::STPMacroValueDictionary&
	SuperTerrainPlus::STPRealism::STPShaderManager::STPShaderSource::STPMacroValueDictionary::operator()(const std::string& macro, T&& value) {
	using std::disjunction_v;
	using std::is_same;

	if constexpr (disjunction_v<
		is_same<T, std::string>, 
		is_same<T, std::string_view>, 
		is_same<T, char*>>) {
		//target is a string, can be inserted directly
		this->Macro.try_emplace(macro, std::forward<T>(value));
	}
	else {
		//value is not a string, needs to be converted first
		this->Macro.try_emplace(macro, std::to_string(std::forward<T>(value)));
	}
	return *this;
}

#endif//_STP_SHADER_MANAGER_H_