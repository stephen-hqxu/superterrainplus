#include <SuperRealism+/Object/STPProgramManager.h>

#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

#include <sstream>

using std::string;
using std::unique_ptr;
using std::stringstream;

using std::endl;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPProgramManager::STPProgramDeleter::operator()(STPOpenGL::STPuint program) const {
	glDeleteProgram(program);
}

STPProgramManager::STPProgramManager() : Program(glCreateProgram()) {

}

void STPProgramManager::resetStatus() {
	this->Linked = false;
	this->Valid = false;

	//make sure old logs are cleared
	this->Log.clear();
}

STPProgramManager& STPProgramManager::attach(const STPShaderManager& shader) {
	//attach shader to the program
	const GLuint shaderID = *shader;

	const bool compilation_err = !static_cast<bool>(*shader),
		repeat_err = this->AttachedShader.find(shaderID) != this->AttachedShader.cend();
	if (compilation_err || repeat_err) {
		//some error occurs
		stringstream msg;
		msg << '(' << shaderID << ',' << shader.Type << ")::";

		if (compilation_err) {
			msg << "Unusable shader";
		}
		else if (repeat_err) {
			msg << "Shader has been attached to this program previously";
		}
		msg << endl;

		//throw error
		throw STPException::STPGLError(msg.str().c_str());
	}
	//no error? good
	glAttachShader(this->Program.get(), shaderID);
	this->AttachedShader.emplace(shaderID, shader.Type);

	return *this;
}

bool STPProgramManager::detach(STPOpenGL::STPenum type) {
	const auto detaching = this->AttachedShader.find(type);
	if (detaching == this->AttachedShader.cend()) {
		//shader does not exist
		return false;
	}

	//shader found
	glDetachShader(this->Program.get(), detaching->second);
	//remove from registry
	this->AttachedShader.erase(detaching);
	return true;
}

void STPProgramManager::clear() {
	for (const auto [shader, type] : this->AttachedShader) {
		glDetachShader(this->Program.get(), shader);
	}
	this->AttachedShader.clear();
	this->resetStatus();
}

void STPProgramManager::separable(bool separable) {
	glProgramParameteri(this->Program.get(), GL_PROGRAM_SEPARABLE, separable ? GL_TRUE : GL_FALSE);
}

void STPProgramManager::finalise() {
	//reset old status, because there are two different flags
	//if the first flag throws error, the second flag should be false but not old value
	this->resetStatus();

	//link
	glLinkProgram(this->Program.get());
	//get log
	GLint logLength;
	glGetProgramiv(this->Program.get(), GL_INFO_LOG_LENGTH, &logLength);
	//get any log
	if (logLength > 0) {
		//shader compilation has log
		this->Log.resize(logLength);
		glGetProgramInfoLog(this->Program.get(), logLength, NULL, this->Log.data());
	}

	auto handle_error = [pro = this->Program.get(), &log = this->Log](GLenum status_request, bool& final_status) -> void {
		//link status error handling
		GLint status;
		glGetProgramiv(pro, status_request, &status);
		//store
		final_status = status == GL_TRUE ? true : false;

		if (!final_status) {
			//error for this stage
			throw STPException::STPGLError(log.c_str());
		}
	};

	//link status error handling
	handle_error(GL_LINK_STATUS, this->Linked);
	//validation checks if the program can be used as a GL application
	handle_error(GL_VALIDATE_STATUS, this->Valid);
}

SuperTerrainPlus::STPOpenGL::STPint STPProgramManager::uniformLocation(const char* uni) const {
	return glGetUniformLocation(this->Program.get(), uni);
}

const string& STPProgramManager::lastLog() const {
	return this->Log;
}

STPProgramManager::operator bool() const {
	return this->Linked && this->Valid;
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