#include <SuperRealism+/Object/STPShaderManager.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//GLAD
#include <glad/glad.h>

//System
#include <sstream>

using std::string;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::vector;
using std::unordered_map;

using std::endl;

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

//TODO: C++20 template lambda
template<class D, size_t S>
static void readSource(D& dict, const char(&pathname)[S]) {
	using namespace SuperTerrainPlus;

	const auto filename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, pathname, ".glsl").data();
	dict[string(pathname) + ".glsl"] = *STPFile(filename);
}

//(include pathname, include source)
static auto mShaderIncludeRegistry = [] {
	unordered_map<string, string> reg;
	//initialise super realism + system include headers
	readSource(reg, "/Common/STPCameraInformation");
	readSource(reg, "/Common/STPAtmosphericScattering");

	return reg;
}();

void STPShaderManager::STPShaderDeleter::operator()(STPOpenGL::STPuint shader) const {
	glDeleteShader(shader);
}

STPShaderManager::STPShaderManager(STPOpenGL::STPenum type) : Shader(glCreateShader(type)), Type(type) {
	
}

bool STPShaderManager::include(const string& name, const string& source) {
	return mShaderIncludeRegistry.try_emplace(name, source).second;
}

bool STPShaderManager::uninclude(const string& name) {
	return mShaderIncludeRegistry.erase(name) == 1ull;
}

void STPShaderManager::cache(const string& source) {
	this->SourceCache = source;
}

unsigned int STPShaderManager::defineMacro(const STPMacroValueDictionary& dictionary) {
	if (this->SourceCache.empty()) {
		//do nothing for empty string
		return 0u;
	}
	constexpr static char DefineIdentifier[] = "#define";
	constexpr static char MacroIdentifier[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_";

	unsigned int macroReplaced = 0u;
	istringstream original_src(this->SourceCache);
	ostringstream output_src;
	string line, macroName;
	//read line by line
	while (std::getline(original_src, line)) {
		if (line.rfind(DefineIdentifier, 0ull) == 0ull) {
			//a define directive is found
			//for accuracy, we only care about #define starting at the first line
			const size_t macroStart = line.find(' ', sizeof(DefineIdentifier) / sizeof(char) - 1ull) + 1ull;
			if (macroStart != string::npos) {
				//get macro name
				const size_t macroEnd = line.find_first_not_of(MacroIdentifier, macroStart);
				//substr will clamp the size so no worries about npos
				macroName = line.substr(macroStart, macroEnd - macroStart);

				//get macro value
				auto dict_it = dictionary.Macro.find(macroName);
				if (dict_it != dictionary.Macro.cend()) {
					//found, compose new line
					output_src << DefineIdentifier << ' ' << macroName << ' ' << dict_it->second << endl;

					macroReplaced++;
					continue;
				}
			}
		}
		
		//insert line into a new cache
		output_src << line << endl;
	}
	//copy stream to cache
	this->SourceCache = output_src.str();

	return macroReplaced;
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
		//check if shader include is supported
		if (!GLAD_GL_ARB_shading_language_include) {
			throw STPException::STPUnsupportedFunctionality("The current rendering context does not support ARB_shading_language_include");
		}

		const size_t pathCount = include.size();
		//build the path information
		vector<const char*> pathStr;
		vector<GLint> pathLength;
		pathStr.reserve(pathCount);
		pathLength.reserve(pathCount);
		for (const auto& path : include) {
			//check if path exists as named string
			if (!glIsNamedStringARB(static_cast<GLint>(path.size()), path.c_str())) {
				//see if this path has been registered
				auto reg_it = mShaderIncludeRegistry.find(path);
				if (reg_it == mShaderIncludeRegistry.cend()) {
					//not registered, we can't do anything
					stringstream msg;
					msg << "Include path \'" << path << "\' is not registered with shader include and no associated source code can be found";
					throw STPException::STPMemoryError(msg.str().c_str());
				}

				const auto& [reg_path, reg_src] = *reg_it;
				//try to add the named string to GL virtual include system
				glNamedStringARB(GL_SHADER_INCLUDE_ARB, static_cast<GLint>(reg_path.size()), reg_path.c_str(), static_cast<GLint>(reg_src.size()), reg_src.c_str());
			}

			pathStr.emplace_back(path.c_str());
			pathLength.emplace_back(static_cast<GLint>(path.size()));
		}


		glCompileShaderIncludeARB(this->Shader.get(), static_cast<GLsizei>(pathCount), pathStr.data(), pathLength.data());
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

const string& STPShaderManager::operator()(const STPShaderIncludePath& include) {
	if (this->SourceCache.empty()) {
		throw STPException::STPMemoryError("There is no source code cached to this shader manager");
	}

	return  this->operator()(this->SourceCache, include);
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