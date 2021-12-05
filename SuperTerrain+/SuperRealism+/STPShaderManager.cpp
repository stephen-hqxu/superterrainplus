#include <SuperRealism+/Object/STPShaderManager.h>

//GLAD
#include <glad/glad.h>

using std::string;
using std::vector;

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPShaderManager::STPShaderDeleter::operator()(GLuint shader) const {
	glDeleteShader(shader);
}

STPShaderManager::STPShaderManager(const string& source, STPOpenGL::STPenum type) : STPShaderManager(vector<const char*>{ source.c_str() }, type) {

}

STPShaderManager::STPShaderManager(const vector<const char*>& source, STPOpenGL::STPenum type) :
	Shader(glCreateShader(type)), Type(type) {
	//attach source code to the shader
	//std::string makes sure string is null-terminated, so we can pass NULL as the code length
	glShaderSource(this->Shader.get(), source.size(), source.data(), NULL);

	//try to compile it
	glCompileShader(this->Shader.get());
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
		return;
	}
	//no log
}

const string& STPShaderManager::getLog() const {
	return this->Log;
}

STPShaderManager::operator bool() const {
	return this->Valid;
}

SuperTerrainPlus::STPOpenGL::STPuint STPShaderManager::operator*() const {
	return this->Shader.get();
}