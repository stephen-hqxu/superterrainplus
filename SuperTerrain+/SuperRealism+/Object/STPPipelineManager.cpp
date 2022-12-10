#include <SuperRealism+/Object/STPPipelineManager.h>

//Log
#include <SuperRealism+/Utility/STPLogHandler.hpp>

//GLAD
#include <glad/glad.h>

#include <algorithm>

using std::string_view;
using std::unique_ptr;
using std::initializer_list;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPPipelineManager::STPPipelineDeleter::operator()(const STPOpenGL::STPuint pipeline) const noexcept {
	glDeleteProgramPipelines(1u, &pipeline);
}

inline static GLuint createOnePipeline() noexcept {
	GLuint pipeline;
	glCreateProgramPipelines(1u, &pipeline);
	return pipeline;
}

STPPipelineManager::STPPipelineManager(const STPPipelineStage* const stage_program, const size_t count) : Pipeline(createOnePipeline()) {
	const GLuint pipeline = this->Pipeline.get();
	//stage all programs
	std::for_each_n(stage_program, count, [pipeline](const auto s_p) {
		const auto [stage, program] = s_p;
		glUseProgramStages(pipeline, stage, *(*program));
	});

	unique_ptr<char[]> log;
	//check for log
	GLint logLength;
	glGetProgramPipelineiv(pipeline, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		log = make_unique<char[]>(logLength);
		glGetProgramPipelineInfoLog(pipeline, logLength, NULL, log.get());
	}

	STPLogHandler::handle(string_view(log.get(), logLength));
}

STPPipelineManager::STPPipelineManager(const initializer_list<const STPPipelineStage> stage_program) :
	STPPipelineManager(std::data(stage_program), stage_program.size()) {

}

SuperTerrainPlus::STPOpenGL::STPuint STPPipelineManager::operator*() const noexcept {
	return this->Pipeline.get();
}

void STPPipelineManager::bind() const noexcept {
	glBindProgramPipeline(this->Pipeline.get());
}

void STPPipelineManager::unbind() noexcept {
	glBindProgramPipeline(0);
}