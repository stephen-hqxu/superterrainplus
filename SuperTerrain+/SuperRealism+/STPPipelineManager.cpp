#include <SuperRealism+/Object/STPPipelineManager.h>

//GLAD
#include <glad/glad.h>

using std::string_view;
using std::optional;

using std::make_unique;
using std::make_optional;

using namespace SuperTerrainPlus::STPRealism;

inline static GLuint createOnePipeline() {
	GLuint pipeline;
	glCreateProgramPipelines(1u, &pipeline);
	return pipeline;
}

STPPipelineManager::STPPipelineManager(const STPShaderSelection& stages) : Pipeline(createOnePipeline()) {
	//assign shader stages
	for (const auto [bit, program] : stages) {
		glUseProgramStages(this->Pipeline, bit, program->Program);
	}

	//check for log
	GLint logLength;
	glGetProgramPipelineiv(this->Pipeline, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		this->Log = make_unique<char[]>(logLength);
		glGetProgramPipelineInfoLog(this->Pipeline, logLength, NULL, this->Log.get());
	}
}

STPPipelineManager::~STPPipelineManager() {
	glDeleteProgramPipelines(1u, &this->Pipeline);
}

optional<string_view> STPPipelineManager::getLog() const {
	return this->Log ? make_optional<string_view>(string_view(this->Log.get())) : optional<string_view>();
}

void STPPipelineManager::bind() const {
	glBindProgramPipeline(this->Pipeline);
}

void STPPipelineManager::unbind() {
	glBindProgramPipeline(0);
}