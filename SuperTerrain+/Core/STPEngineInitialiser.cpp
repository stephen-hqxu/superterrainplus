#include <SuperTerrain+/STPEngineInitialiser.h>

//GLAD
#include <glad/glad.h>
//CUDA
#include <cuda_runtime.h>
//Compatibility
#include <SuperTerrain+/STPOpenGL.h>
#include <SuperTerrain+/STPSQLite.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Utility/STPDatabaseErrorHandler.hpp>

#include <SuperTerrain+/Exception/STPGLError.h>

#include <type_traits>

using namespace SuperTerrainPlus;

using std::is_same;
using std::conjunction_v;

//Compatibility checking
static_assert(conjunction_v<
	is_same<STPOpenGL::STPuint, GLuint>,
	is_same<STPOpenGL::STPint, GLint>,
	is_same<STPOpenGL::STPsizei, GLsizei>,
	is_same<STPOpenGL::STPenum, GLenum>,
	is_same<STPOpenGL::STPbitfield, GLbitfield>,
	is_same<STPOpenGL::STPboolean, GLboolean>,
	is_same<STPOpenGL::STPfloat, GLfloat>,
	is_same<STPOpenGL::STPuint64, GLuint64>,
	is_same<STPOpenGL::STPintptr, GLintptr>,
	is_same<STPOpenGL::STPsizeiptr, GLsizeiptr>
>, "OpenGL specification is no longer compatible with SuperTerrain+, please contact the maintainer.");

void STPEngineInitialiser::initialise(const int device, const STPGLProc gl_process) {
	//CUDA
	STP_CHECK_CUDA(cudaSetDevice(device));
	//init context in case the first call is CUDA driver API
	STP_CHECK_CUDA(cudaFree(0));
	//setup
	STP_CHECK_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	STP_CHECK_CUDA(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

	//SQLite
	STP_CHECK_SQLITE3(sqlite3_initialize());

	//GL
	if (gl_process == nullptr) {
		//no initialisation to GL context
		return;
	}

	if (!gladLoadGLLoader(gl_process)) {
		throw STPException::STPGLError("Unable to setup GL context");
	}
}

int STPEngineInitialiser::architecture(const int device) {
	int major, minor;
	STP_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	STP_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

	return major * 10 + minor;
}