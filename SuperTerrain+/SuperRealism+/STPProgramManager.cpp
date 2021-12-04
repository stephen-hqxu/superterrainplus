#include <SuperRealism+/Object/STPProgramManager.h>

//GLAD
#include <glad/glad.h>

using std::string;
using std::string_view;
using std::list;
using std::unique_ptr;
using std::optional;

using std::make_unique;
using std::make_optional;

using namespace SuperTerrainPlus::STPRealism;

STPProgramManager::STPProgramManager(const STPShaderGroup& shader_group, const STPProgramParameteri& parameter) : Program(glCreateProgram()) {
	//apply program parameters if any
	for (const auto [pname, value] : parameter) {
		glProgramParameteri(this->Program, pname, value);
	}

	//attach all shaders to the program
	for (const auto* shader : shader_group) {
		if (!static_cast<bool>(*shader)) {
			//skip any failed shader
			continue;
		}
		glAttachShader(this->Program, shader->Shader);
	}
	//link
	glLinkProgram(this->Program);

	auto handle_error = [pro = this->Program](GLenum status_request, unique_ptr<char[]>& log, bool& final_status) -> void {
		//link status error handling
		GLint logLength, status;
		glGetProgramiv(pro, GL_INFO_LOG_LENGTH, &logLength);
		glGetProgramiv(pro, status_request, &status);
		//store
		final_status = status == GL_TRUE ? true : false;
		//get any log
		if (logLength > 0) {
			//shader compilation has log
			log = make_unique<char[]>(logLength);
			glGetProgramInfoLog(pro, logLength, NULL, log.get());
			return;
		}
	};

	//link status error handling
	handle_error(GL_LINK_STATUS, this->LinkLog, this->Linked);
	//validation checks if the program can be used as a GL application
	handle_error(GL_VALIDATE_STATUS, this->ValidationLog, this->Valid);
}

STPProgramManager::~STPProgramManager() {
	glDeleteProgram(this->Program);
	//deletion of program also detaches all shaders.
}

SuperTerrainPlus::STPOpenGL::STPint STPProgramManager::uniformLocation(const char* uni) const {
	return glGetUniformLocation(this->Program, uni);
}

optional<string_view> STPProgramManager::getLog(STPLogType log_type) const {
	const char* log;

	//if there is no such log type, unique_ptr should return nullptr
	switch (log_type) {
	case STPLogType::Link: log = this->LinkLog.get();
		break;
	case STPLogType::Validation: log = this->ValidationLog.get();
		break;
	default:
		//impossible
		break;
	}
	return log != nullptr ? make_optional<string_view>(string_view(log)) : optional<string_view>();
}

STPProgramManager::operator bool() const {
	return this->Linked && this->Valid;
}

void STPProgramManager::use() const {
	glUseProgram(this->Program);
}

void STPProgramManager::unuse() {
	glUseProgram(0);
}