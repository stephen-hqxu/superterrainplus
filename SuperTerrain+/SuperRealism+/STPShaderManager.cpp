#include <SuperRealism+/Object/STPShaderManager.h>

//GLAD
#include <glad/glad.h>

using std::string;
using std::string_view;
using std::vector;
using std::optional;

using std::make_unique;
using std::make_optional;

using namespace SuperTerrainPlus::STPRealism;

STPShaderManager::STPShaderManager(const string& source, STPOpenGL::STPenum type) : STPShaderManager(vector<const char*>{ source.c_str() }, type) {

}

STPShaderManager::STPShaderManager(const vector<const char*>& source, STPOpenGL::STPenum type) :
	Shader(glCreateShader(type)), Type(type) {
	//attach source code to the shader
	//std::string makes sure string is null-terminated, so we can pass NULL as the code length
	glShaderSource(this->Shader, source.size(), source.data(), NULL);

	//try to compile it
	glCompileShader(this->Shader);
	//retrieve any log
	GLint logLength, status;
	glGetShaderiv(this->Shader, GL_INFO_LOG_LENGTH, &logLength);
	glGetShaderiv(this->Shader, GL_COMPILE_STATUS, &status);
	
	//store information
	this->Valid = status == GL_TRUE ? true : false;
	if (logLength > 0) {
		//shader compilation has log
		this->Log = make_unique<char[]>(logLength);
		glGetShaderInfoLog(this->Shader, logLength, NULL, this->Log.get());
		return;
	}
	//no log? The log variable will be null
}

STPShaderManager::~STPShaderManager() {
	glDeleteShader(this->Shader);
	//flag the shader for deletion.
	//as long as the current shader is not attached to any program, it will be actually deleted.
}

optional<string_view> STPShaderManager::getLog() const {
	return this->Log ? make_optional<string_view>(string_view(this->Log.get())) : optional<string_view>();
}

STPShaderManager::operator bool() const {
	return this->Valid;
}