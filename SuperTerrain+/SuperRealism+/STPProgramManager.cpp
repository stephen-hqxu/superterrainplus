#include <SuperRealism+/Object/STPProgramManager.h>

#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//Log Utility
#include <SuperRealism+/Utility/STPLogHandler.hpp>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

#include <sstream>

using std::string;
using std::unique_ptr;
using std::stringstream;

using std::endl;
using std::make_unique;

using glm::ivec3;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

void STPProgramManager::STPProgramDeleter::operator()(STPOpenGL::STPuint program) const {
	glDeleteProgram(program);
}

STPProgramManager::STPProgramManager() : Program(glCreateProgram()), isComputeProgram(false) {

}

inline STPProgramManager::STPShaderDatabase::iterator STPProgramManager::detachByIterator(STPShaderDatabase::iterator it) {
	//shader found
	glDetachShader(this->Program.get(), it->second);
	//remove from registry
	return this->AttachedShader.erase(it);
}

STPProgramManager& STPProgramManager::attach(const STPShaderManager& shader) {
	//attach shader to the program
	const GLenum shaderType = shader.Type;
	const GLuint shaderRef = *shader;

	const bool compilation_err = !static_cast<bool>(*shader),
		repeat_err = this->AttachedShader.find(shaderType) != this->AttachedShader.cend();
	if (compilation_err || repeat_err) {
		//some error occurs
		stringstream msg;
		msg << '(' << shaderType << ',' << shaderRef << ")::";

		if (compilation_err) {
			msg << "Unusable shader";
		}
		else if (repeat_err) {
			msg << "This type of shader has been attached to this program previously";
		}
		msg << endl;

		//throw error
		throw STPException::STPGLError(msg.str().c_str());
	}
	//no error? good
	glAttachShader(this->Program.get(), shaderRef);
	this->AttachedShader.emplace(shaderType, shaderRef);

	return *this;
}

bool STPProgramManager::detach(STPOpenGL::STPenum type) {
	const auto detaching = this->AttachedShader.find(type);
	if (detaching == this->AttachedShader.cend()) {
		//shader does not exist
		return false;
	}

	this->detachByIterator(detaching);
	return true;
}

void STPProgramManager::clear() {
	for (const auto [type, shader] : this->AttachedShader) {
		glDetachShader(this->Program.get(), shader);
	}
	this->AttachedShader.clear();
}

void STPProgramManager::separable(bool separable) {
	glProgramParameteri(this->Program.get(), GL_PROGRAM_SEPARABLE, separable ? GL_TRUE : GL_FALSE);
}

void STPProgramManager::finalise() {
	//link
	glLinkProgram(this->Program.get());
	//get log
	GLint logLength;
	glGetProgramiv(this->Program.get(), GL_INFO_LOG_LENGTH, &logLength);

	string log;
	//get any log
	if (logLength > 0) {
		//shader compilation has log
		log.resize(logLength);
		glGetProgramInfoLog(this->Program.get(), logLength, NULL, log.data());
	}

	auto handle_error = [pro = this->Program.get(), &log](GLenum status_request) -> void {
		//link status error handling
		GLint status;
		glGetProgramiv(pro, status_request, &status);
		//store
		const bool valid = status == GL_TRUE ? true : false;

		if (!valid) {
			//error for this stage
			throw STPException::STPGLError(log.c_str());
		}
	};

	//link status error handling
	handle_error(GL_LINK_STATUS);
	//validation checks if the program can be used as a GL application
	handle_error(GL_VALIDATE_STATUS);

	STPLogHandler::ActiveLogHandler->handle(std::move(log));

	//check the status of the program
	this->isComputeProgram = this->AttachedShader.count(GL_COMPUTE_SHADER) > 0u;

	//detach all shaders after a successful linking so shader can be deleted
	for (auto it = this->AttachedShader.begin(); it != this->AttachedShader.end(); it = this->detachByIterator(it)) {
		//I am empty
	}
}

SuperTerrainPlus::STPOpenGL::STPint STPProgramManager::uniformLocation(const char* uni) const {
	return glGetUniformLocation(this->Program.get(), uni);
}

ivec3 STPProgramManager::workgroupSize() const {
	if (!this->isComputeProgram) {
		throw STPException::STPUnsupportedFunctionality("The target program is not a compute shader program");
	}

	//query the size
	ivec3 size;
	glGetProgramiv(this->Program.get(), GL_COMPUTE_WORK_GROUP_SIZE, value_ptr(size));
	return size;
}

SuperTerrainPlus::STPOpenGL::STPuint STPProgramManager::operator*() const {
	return this->Program.get();
}

void STPProgramManager::use() const {
	glUseProgram(this->Program.get());
}

void STPProgramManager::unuse() {
	glUseProgram(0);
}