#include <SuperRealism+/Utility/STPDebugCallback.h>

//Error
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

using std::ostream;

using namespace SuperTerrainPlus::STPRealism;

//Indicate if debug output is enabled.
static bool mEnabled = false;

static void defaultDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
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

void STPDebugCallback::enable() {
	if (!support()) {
		//Does not support
		throw STPException::STPUnsupportedFunctionality("The current rendering platform does not support GL debug output");
	}
	if (mEnabled) {
		return;
	}

	//only enable if the GPU has support to it
	glEnable(GL_DEBUG_OUTPUT);
	mEnabled = true;
}

void STPDebugCallback::disable() {
	if (!mEnabled) {
		return;
	}

	glDisable(GL_DEBUG_OUTPUT);
	mEnabled = false;
}

bool STPDebugCallback::isEnabled() {
	return mEnabled;
}

inline static void checkEnable() {
	if (!mEnabled) {
		//debug callback not enabled
		throw SuperTerrainPlus::STPException::STPGLError("Debug callback was not initialised");
	}
}

void STPDebugCallback::enableAsyncCallback(ostream& stream) {
	checkEnable();

	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	//use the default message callback
	glDebugMessageCallback(&defaultDebugOutput, &stream);
}

void STPDebugCallback::disableAsyncCallback() {
	checkEnable();

	glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
}