#include <SuperRealism+/Object/STPProgramManager.h>

//GLAD
#include <glad/glad.h>

using std::string;
using std::list;
using std::unique_ptr;

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPProgramManager::STPProgramDeleter::operator()(GLuint program) const {
	glDeleteProgram(program);
}

STPProgramManager::STPProgramManager(const STPShaderGroup& shader_group, const STPProgramParameteri& parameter) : Program(glCreateProgram()) {
	//apply program parameters if any
	for (const auto [pname, value] : parameter) {
		glProgramParameteri(this->Program.get(), pname, value);
	}

	//attach all shaders to the program
	for (const auto* shader : shader_group) {
		if (!static_cast<bool>(*shader)) {
			//skip any failed shader
			continue;
		}
		glAttachShader(this->Program.get(), **shader);
	}
	//link
	glLinkProgram(this->Program.get());

	auto handle_error = [pro = this->Program.get()](GLenum status_request, string& log, bool& final_status) -> void {
		//link status error handling
		GLint logLength, status;
		glGetProgramiv(pro, GL_INFO_LOG_LENGTH, &logLength);
		glGetProgramiv(pro, status_request, &status);
		//store
		final_status = status == GL_TRUE ? true : false;
		//get any log
		if (logLength > 0) {
			//shader compilation has log
			log.resize(logLength);
			glGetProgramInfoLog(pro, logLength, NULL, log.data());
			return;
		}
	};

	//link status error handling
	handle_error(GL_LINK_STATUS, this->LinkLog, this->Linked);
	//validation checks if the program can be used as a GL application
	handle_error(GL_VALIDATE_STATUS, this->ValidationLog, this->Valid);
}

SuperTerrainPlus::STPOpenGL::STPint STPProgramManager::uniformLocation(const char* uni) const {
	return glGetUniformLocation(this->Program.get(), uni);
}

const string& STPProgramManager::getLog(STPLogType log_type) const {
	static string Dummy = "If you get this message, there is something wrong with the terrain engine, report to the maintainer.";

	//if there is no such log type, unique_ptr should return nullptr
	switch (log_type) {
	case STPLogType::Link: return this->LinkLog;
	case STPLogType::Validation: return this->ValidationLog;
	default: return Dummy;
	}
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