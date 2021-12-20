#include <SuperRealism+/Utility/STPDebugCallback.h>

//GLAD
#include <glad/glad.h>

using std::ostream;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

static void defaultDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei, const GLchar* message, const void* userParam) {
	//string convertion
	static auto getSourceStr = [](GLenum source) constexpr -> char* {
		switch (source) {
		case GL_DEBUG_SOURCE_API: return "API";
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "WINDOW SYSTEM";
		case GL_DEBUG_SOURCE_SHADER_COMPILER: return "SHADER COMPILER";
		case GL_DEBUG_SOURCE_THIRD_PARTY: return "THIRD PARTY";
		case GL_DEBUG_SOURCE_APPLICATION: return "APPLICATION";
		case GL_DEBUG_SOURCE_OTHER: return "OTHER";
		default: return "NULL";
		}
	};
	static auto getTypeStr = [](GLenum type) constexpr -> char* {
		switch (type) {
		case GL_DEBUG_TYPE_ERROR: return "ERROR";
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DEPRECATED_BEHAVIOR";
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "UNDEFINED_BEHAVIOR";
		case GL_DEBUG_TYPE_PORTABILITY: return "PORTABILITY";
		case GL_DEBUG_TYPE_PERFORMANCE: return "PERFORMANCE";
		case GL_DEBUG_TYPE_MARKER: return "MARKER";
		case GL_DEBUG_TYPE_OTHER: return "OTHER";
		default: return "NULL";
		}
	};
	static auto getSeverityStr = [](GLenum severity) constexpr -> char* {
		switch (severity) {
		case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFICATION";
		case GL_DEBUG_SEVERITY_LOW: return "LOW";
		case GL_DEBUG_SEVERITY_MEDIUM: return "MEDIUM";
		case GL_DEBUG_SEVERITY_HIGH: return "HIGH";
		default: return "NULL";
		}
	};
	
	using std::endl;
	//user parameter has a stream
	ostream& stream = *const_cast<ostream*>(reinterpret_cast<const ostream*>(userParam));
	stream << getSourceStr(source) << '(' << getTypeStr(type) << "::" << getSeverityStr(severity) << "):" << id << ':' << endl;;
	stream << message << endl;
}

int STPDebugCallback::support() {
	return GLAD_GL_ARB_debug_output;
}

void STPDebugCallback::registerAsyncCallback(ostream& stream) {
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	//use the default message callback
	glDebugMessageCallback(&defaultDebugOutput, &stream);
}