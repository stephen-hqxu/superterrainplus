//INLINE DEFINITION FOR PROGRAM MANAGER, PLEASE DO NOT INCLUDE MANUALLY
#ifdef _STP_PROGRAM_MANAGER_H_

#include <tuple>
#include <functional>

template<typename Uni, typename... Arg>
SuperTerrainPlus::STPRealism::STPProgramManager& SuperTerrainPlus::STPRealism::STPProgramManager::uniform(
	Uni&& uniform_function, const char* uni, Arg&&... args) noexcept {
	return this->uniform(uniform_function, this->uniformLocation(uni), args...);
}

template<typename Uni, typename... Arg>
SuperTerrainPlus::STPRealism::STPProgramManager& SuperTerrainPlus::STPRealism::STPProgramManager::uniform(
	Uni&& uniform_function, STPOpenGL::STPint location, Arg&&... args) noexcept {
	using std::apply;
	using std::make_tuple;

	//pack arguments into a tuple
	//call the uniform
	apply(std::forward<Uni>(uniform_function), make_tuple(this->Program.get(), location, std::forward<Arg>(args)...));

	return *this;
}

#endif//_STP_PROGRAM_MANAGER_H_