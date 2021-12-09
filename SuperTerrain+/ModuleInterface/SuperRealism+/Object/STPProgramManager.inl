//INLINE DEFINITION FOR PROGRAM MANAGER, PLEASE DO NOT INCLUDE MANUALLY

#ifdef _STP_PROGRAM_MANAGER_H_

template<typename Uni, typename... Arg>
SuperTerrainPlus::STPRealism::STPProgramManager& SuperTerrainPlus::STPRealism::STPProgramManager::uniform
	(Uni&& uniform_function, const char* uni, Arg&&... args) {
	using std::tuple;
	using std::apply;
	using std::make_tuple;

	//pack arguments into a tuple
	const tuple<STPOpenGL::STPuint, STPOpenGL::STPint, Arg...> uniformArg = 
		make_tuple(this->Program.get(), this->uniformLocation(uni), std::forward<Arg>(args)...);
	//call the uniform
	apply(uniform_function, uniformArg);

	return *this;
}

#endif//_STP_PROGRAM_MANAGER_H_