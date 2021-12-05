#include <SuperRealism+/Object/STPPipelineManager.h>

//GLAD
#include <glad/glad.h>

using std::string;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPPipelineManager::STPPipelineDeleter::operator()(GLuint pipeline) const {
	glDeleteProgramPipelines(1u, &pipeline);
}

inline static GLuint createOnePipeline() {
	GLuint pipeline;
	glCreateProgramPipelines(1u, &pipeline);
	return pipeline;
}

STPPipelineManager::STPPipelineManager(const STPShaderSelection& stages) : Pipeline(createOnePipeline()) {
	//assign shader stages
	for (const auto [bit, program] : stages) {
		glUseProgramStages(this->Pipeline.get(), bit, **program);
	}

	//check for log
	GLint logLength;
	glGetProgramPipelineiv(this->Pipeline.get(), GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		this->Log.resize(logLength);
		glGetProgramPipelineInfoLog(this->Pipeline.get(), logLength, NULL, this->Log.data());
	}
}

SuperTerrainPlus::STPOpenGL::STPuint STPPipelineManager::operator*() const {
	return this->Pipeline.get();
}

const string& STPPipelineManager::getLog() const {
	return this->Log;
}

void STPPipelineManager::bind() const {
	glBindProgramPipeline(this->Pipeline.get());
}

void STPPipelineManager::unbind() {
	glBindProgramPipeline(0);
}