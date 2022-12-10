#include <SuperRealism+/Utility/STPDebugCallback.h>

#include <SuperTerrain+/Exception/STPInvalidEnum.h>

//GLAD
#include <glad/glad.h>

#include <string>
#include <string_view>

using std::to_string;
using std::string_view;
using std::ostream;
using std::endl;

using namespace SuperTerrainPlus::STPRealism;

ostream& STPDebugCallback::print(const STPOpenGL::STPenum source, const STPOpenGL::STPenum type, const STPOpenGL::STPuint id,
	const STPOpenGL::STPenum severity, const STPOpenGL::STPsizei length, const char* const message, ostream& stream) {
	//string conversion
	static constexpr auto getSourceStr = [](const GLenum source) constexpr -> const char* {
		switch (source) {
		case GL_DEBUG_SOURCE_API: return "API";
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "WINDOW SYSTEM";
		case GL_DEBUG_SOURCE_SHADER_COMPILER: return "SHADER COMPILER";
		case GL_DEBUG_SOURCE_THIRD_PARTY: return "THIRD PARTY";
		case GL_DEBUG_SOURCE_APPLICATION: return "APPLICATION";
		case GL_DEBUG_SOURCE_OTHER: return "OTHER";
		default: throw STP_INVALID_STRING_ENUM_CREATE(to_string(source), "GL Debug Source");
		}
	};
	static constexpr auto getTypeStr = [](const GLenum type) constexpr -> const char* {
		switch (type) {
		case GL_DEBUG_TYPE_ERROR: return "ERROR";
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DEPRECATED_BEHAVIOR";
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "UNDEFINED_BEHAVIOR";
		case GL_DEBUG_TYPE_PORTABILITY: return "PORTABILITY";
		case GL_DEBUG_TYPE_PERFORMANCE: return "PERFORMANCE";
		case GL_DEBUG_TYPE_MARKER: return "MARKER";
		case GL_DEBUG_TYPE_OTHER: return "OTHER";
		default: throw STP_INVALID_STRING_ENUM_CREATE(to_string(type), "GL Debug Type");
		}
	};
	static constexpr auto getSeverityStr = [](const GLenum severity) constexpr -> const char* {
		switch (severity) {
		case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFICATION";
		case GL_DEBUG_SEVERITY_LOW: return "LOW";
		case GL_DEBUG_SEVERITY_MEDIUM: return "MEDIUM";
		case GL_DEBUG_SEVERITY_HIGH: return "HIGH";
		default: throw STP_INVALID_STRING_ENUM_CREATE(to_string(severity), "GL Debug Severity");
		}
	};

	//the GL specification doesn't guarantee the message is null-terminated
	stream << getSourceStr(source) << '(' << getTypeStr(type) << "::" << getSeverityStr(severity) << "):" << id << ':' << endl;
	stream << string_view(message, length) << endl;
	return stream;
}