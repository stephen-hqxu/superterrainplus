#include <SuperRealism+/Object/STPShaderManager.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

using std::string;
using std::vector;

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPShaderManager::STPShaderDeleter::operator()(STPOpenGL::STPuint shader) const {
	glDeleteShader(shader);
}

STPShaderManager::STPShaderManager(STPOpenGL::STPenum type) : Shader(glCreateShader(type)), Type(type) {
	
}

const string& STPShaderManager::operator()(const string& source, const STPShaderIncludePath& include) {
	//attach source code to the shader
	//std::string makes sure string is null-terminated, so we can pass NULL as the code length
	const char* const sourceArray = source.c_str();
	const GLint sourceLength = static_cast<GLint>(source.size());
	glShaderSource(this->Shader.get(), 1, &sourceArray, &sourceLength);

	//try to compile it
	if (include.empty()) {
		glCompileShader(this->Shader.get());
	}
	else {
		const size_t pathCount = include.size();
		//build the path information
		vector<const char*> pathStr;
		vector<GLint> pathLength;
		pathStr.reserve(pathCount);
		pathLength.reserve(pathCount);
		for (const auto& path : include) {
			pathStr.emplace_back(path.c_str());
			pathLength.emplace_back(static_cast<GLint>(path.size()));
		}

		glCompileShaderIncludeARB(this->Shader.get(), pathCount, pathStr.data(), pathLength.data());
	}
	//retrieve any log
	GLint logLength, status;
	glGetShaderiv(this->Shader.get(), GL_INFO_LOG_LENGTH, &logLength);
	glGetShaderiv(this->Shader.get(), GL_COMPILE_STATUS, &status);

	//store information
	this->Valid = status == GL_TRUE ? true : false;
	if (logLength > 0) {
		//shader compilation has log
		this->Log.resize(logLength);
		glGetShaderInfoLog(this->Shader.get(), logLength, NULL, this->Log.data());
	}
	else {
		//clear old compilation old, leaving no log at current
		this->Log.clear();
	}

	if (!this->Valid) {
		//compilation error
		throw STPException::STPGLError(this->Log.c_str());
	}
	return this->lastLog();
}

const string& STPShaderManager::lastLog() const {
	return this->Log;
}

STPShaderManager::operator bool() const {
	return this->Valid;
}

SuperTerrainPlus::STPOpenGL::STPuint STPShaderManager::operator*() const {
	return this->Shader.get();
}