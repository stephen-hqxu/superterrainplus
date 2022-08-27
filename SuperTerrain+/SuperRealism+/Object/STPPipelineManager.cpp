#include <SuperRealism+/Object/STPPipelineManager.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>

//Log
#include <SuperRealism+/Utility/STPLogHandler.hpp>

//GLAD
#include <glad/glad.h>

using std::string_view;
using std::unique_ptr;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPPipelineManager::STPPipelineDeleter::operator()(STPOpenGL::STPuint pipeline) const {
	glDeleteProgramPipelines(1u, &pipeline);
}

inline static GLuint createOnePipeline() {
	GLuint pipeline;
	glCreateProgramPipelines(1u, &pipeline);
	return pipeline;
}

STPPipelineManager::STPPipelineManager() : Pipeline(createOnePipeline()) {
	
}

STPPipelineManager& STPPipelineManager::stage(STPOpenGL::STPbitfield stage, const STPProgramManager& program) {
	glUseProgramStages(this->Pipeline.get(), stage, *program);
	return *this;
}

void STPPipelineManager::finalise() {
	unique_ptr<char[]> log;
	//check for log
	GLint logLength;
	glGetProgramPipelineiv(this->Pipeline.get(), GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		log = make_unique<char[]>(logLength);
		glGetProgramPipelineInfoLog(this->Pipeline.get(), logLength, NULL, log.get());
	}

	STPLogHandler::ActiveLogHandler->handle(string_view(log.get(), logLength));
}

SuperTerrainPlus::STPOpenGL::STPuint STPPipelineManager::operator*() const {
	return this->Pipeline.get();
}

void STPPipelineManager::bind() const {
	glBindProgramPipeline(this->Pipeline.get());
}

void STPPipelineManager::unbind() {
	glBindProgramPipeline(0);
}