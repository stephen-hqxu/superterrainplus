#include <SuperTerrain+/STPEngineInitialiser.h>

//GLAD
#include <glad/glad.h>
//CUDA
#include <cuda_runtime.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
//Compatibility
#include <SuperTerrain+/STPOpenGL.h>
#include <SuperTerrain+/STPSQLite.h>

#include <type_traits>

using namespace SuperTerrainPlus;

using std::is_same;
using std::conjunction_v;

//Default state is false, once the engine is initialised it will become true.
static bool GLInit = false;
static bool EngineInit = false;

//Compatibility checking
static_assert(conjunction_v<
	is_same<STPOpenGL::STPenum, GLenum>, 
	is_same<STPOpenGL::STPuint, GLuint>, 
	is_same<STPOpenGL::STPint, GLint>,
	is_same<STPOpenGL::STPbitfield, GLbitfield>,
	is_same<STPOpenGL::STPboolean, GLboolean>,
	is_same<STPOpenGL::STPintptr, GLintptr>
>,
	"OpenGL specification is no longer compatible with SuperTerrain+, please contact the maintainer.");

bool STPEngineInitialiser::initGLcurrent() {
	if (!gladLoadGL()) {
		return false;
	}
	GLInit = true;
	return GLInit;
}

bool STPEngineInitialiser::initGLexplicit(STPglProc process) {
	if (!gladLoadGLLoader(process)) {
		return false;
	}
	GLInit = true;
	return GLInit;
}

void STPEngineInitialiser::init(int device) {
	//CUDA
	STPcudaCheckErr(cudaSetDevice(device));
	//init context in case the first call is CUDA driver API
	STPcudaCheckErr(cudaFree(0));
	//setup
	STPcudaCheckErr(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	STPcudaCheckErr(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

	//SQLite3
	STPsqliteCheckErr(sqlite3_enable_shared_cache(false));

	EngineInit = true;
}

bool STPEngineInitialiser::hasInit() {
	return GLInit && EngineInit;
}