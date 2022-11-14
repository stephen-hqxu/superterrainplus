#include <SuperRealism+/Utility/STPDebugCallback.h>

//GLAD
#include <glad/glad.h>

#include <string_view>

using std::string_view;
using std::ostream;
using std::endl;

using namespace SuperTerrainPlus::STPRealism;

ostream& STPDebugCallback::print(STPOpenGL::STPenum source, STPOpenGL::STPenum type, STPOpenGL::STPuint id,
	STPOpenGL::STPenum severity, STPOpenGL::STPsizei length, const char* message, ostream& stream) {
	//string conversion
	static constexpr auto getSourceStr = [](GLenum source) constexpr -> const char* {
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
	static constexpr auto getTypeStr = [](GLenum type) constexpr -> const char* {
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
	static constexpr auto getSeverityStr = [](GLenum severity) constexpr -> const char* {
		switch (severity) {
		case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFICATION";
		case GL_DEBUG_SEVERITY_LOW: return "LOW";
		case GL_DEBUG_SEVERITY_MEDIUM: return "MEDIUM";
		case GL_DEBUG_SEVERITY_HIGH: return "HIGH";
		default: return "NULL";
		}
	};

	//the GL specification doesn't guarantee the message is null-terminated
	stream << getSourceStr(source) << '(' << getTypeStr(type) << "::" << getSeverityStr(severity) << "):" << id << ':' << endl;
	stream << string_view(message, length) << endl;
	return stream;
}