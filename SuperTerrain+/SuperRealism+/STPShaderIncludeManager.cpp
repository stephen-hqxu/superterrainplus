#include <SuperRealism+/Utility/STPShaderIncludeManager.h>

//Error
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

#include <sstream>

using std::string;
using std::stringstream;

using std::endl;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

int STPShaderIncludeManager::support() {
	return GLAD_GL_ARB_shading_language_include;
}

inline static void checkSupport() {
	if (!STPShaderIncludeManager::support()) {
		throw STPException::STPUnsupportedFunctionality("The current GPU does not support shading language include extension");
	}
}

inline static bool isRegistered(size_t length, const char* path) {
	return glIsNamedStringARB(length, path);
}

bool STPShaderIncludeManager::addSource(const string& path, const string& source) {
	checkSupport();

	//make sure the same path has not been registered previously
	if (exist(path)) {
		//if path duplicates, GL will replace the old content, so we need to avoid this.
		return false;
	}
	//add
	glNamedStringARB(GL_SHADER_INCLUDE_ARB, path.size(), path.data(), source.size(), source.data());
	return true;
}

inline static bool removeSourceImpl(size_t length, const char* path) {
	if (!isRegistered(length, path)) {
		//path is not previously registered
		//OpenGL throws error if path is not found.
		return false;
	}
	//path is valid, remove it
	glDeleteNamedStringARB(length, path);
	return true;
}

bool STPShaderIncludeManager::removeSource(const string& path) {
	checkSupport();

	return removeSourceImpl(path.size(), path.data());
}

bool STPShaderIncludeManager::exist(const std::string& path) {
	checkSupport();

	return isRegistered(path.size(), path.data());
}

STPShaderIncludeManager::STPManagedSourceDeleter::STPManagedSourceDeleter(size_t length) : PathLength(length) {

}

void STPShaderIncludeManager::STPManagedSourceDeleter::operator()(const char* path) const {
	//make sure path name still exists in the file system
	removeSourceImpl(this->PathLength, path);

	//delete the memory of the path itself
	std::default_delete<const char>()(path);
}

STPShaderIncludeManager::STPManagedSource STPShaderIncludeManager::registerRemover(const string& path) {
	checkSupport();

	if (!exist(path)) {
		//not registered
		stringstream msg;
		msg << "Unable to register a remover for \'" << path << "\' because it is not added as a named string previously" << endl;
		throw STPException::STPMemoryError(msg.str().c_str());
	}

	//preapre the path string for managed memory
	//std::string length does not include null terminator, be careful
	const size_t pathLength = path.size(), pathLengthNull = pathLength + 1ull;
	char* name = new char[pathLengthNull];
	//string copy does not include the null
	path.copy(name, pathLength);
	name[pathLength] = '\0';

	return STPManagedSource(name, pathLengthNull);
}