#include <SuperTerrain+/STPEngineInitialiser.h>

//GLAD
#include <glad/glad.h>
//CUDA
#include <cuda_runtime.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
//SQLite3
#include <SuperTerrain+/Utility/STPSQLite.h>

using namespace SuperTerrainPlus;

//Default state is false, once the engine is initialised it will become true.
static bool GLInit = false;
static bool EngineInit = false;

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