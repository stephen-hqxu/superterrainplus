#include <SuperRealism+/Object/STPProgramManager.h>

#include <SuperTerrain+/Exception/STPGLError.h>

//Log Utility
#include <SuperRealism+/Utility/STPLogHandler.hpp>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

#include <sstream>
#include <algorithm>

using std::unique_ptr;
using std::stringstream;
using std::string_view;
using std::initializer_list;

using std::endl;
using std::make_unique;
using std::for_each_n;

using glm::ivec3;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

void STPProgramManager::STPProgramDeleter::operator()(STPOpenGL::STPuint program) const noexcept {
	glDeleteProgram(program);
}

STPProgramManager::STPProgramManager(const STPShaderManager::STPShader* const* shader_ptr, size_t count,
	const STPProgramCompileOption& option) : Program(glCreateProgram()) {
	const GLuint program = this->Program.get();
	//attach all shaders
	for_each_n(shader_ptr, count, [program](const auto& shader) { glAttachShader(program, shader->get()); });
	//set compiler option
	if (option) {
		const auto [separable] = *option;
		//only need to set it to true, because it is false already by default
		if (separable) {
			glProgramParameteri(program, GL_PROGRAM_SEPARABLE, GL_TRUE);
		}
	}
	//link the program
	glLinkProgram(program);
	//clean up, detach all shaders
	for_each_n(shader_ptr, count, [program](const auto& shader) { glDetachShader(program, shader->get()); });

	//status check
	GLint logLength, status;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
	glGetProgramiv(program, GL_LINK_STATUS, &status);

	unique_ptr<char[]> log;
	const bool valid = status == GL_TRUE ? true : false;
	//get log
	if (logLength > 0) {
		//program has log
		log = make_unique<char[]>(logLength);
		glGetProgramInfoLog(program, logLength, nullptr, log.get());
	}
	if (!valid) {
		throw STPException::STPGLError(log.get());
	}
	//write log
	STPLogHandler::ActiveLogHandler->handle(string_view(log.get(), logLength));
}

STPProgramManager::STPProgramManager(initializer_list<const STPShaderManager::STPShader*> shader, const STPProgramCompileOption& option) :
	STPProgramManager(std::data(shader), shader.size(), option) {

}

SuperTerrainPlus::STPOpenGL::STPint STPProgramManager::uniformLocation(const char* uni) const noexcept {
	return glGetUniformLocation(this->Program.get(), uni);
}

ivec3 STPProgramManager::workgroupSize() const noexcept {
	//query the size
	ivec3 size;
	glGetProgramiv(this->Program.get(), GL_COMPUTE_WORK_GROUP_SIZE, value_ptr(size));
	return size;
}

SuperTerrainPlus::STPOpenGL::STPuint STPProgramManager::operator*() const noexcept {
	return this->Program.get();
}

void STPProgramManager::use() const noexcept {
	glUseProgram(this->Program.get());
}

void STPProgramManager::unuse() noexcept {
	glUseProgram(0);
}