#pragma once
#include <SuperTerrain+/STPEngineInitialiser.h>

//GLAD
#include <glad/glad.h>
//CUDA
#include <cuda_runtime.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

using namespace SuperTerrainPlus;

bool STPEngineInitialiser::GLInited = false;
bool STPEngineInitialiser::CUDAInited = false;

bool STPEngineInitialiser::initGLcurrent() {
	if (!gladLoadGL()) {
		return false;
	}
	STPEngineInitialiser::GLInited = true;
	return true;
}

bool STPEngineInitialiser::initGLexplicit(STPglProc process) {
	if (!gladLoadGLLoader(process)) {
		return false;
	}
	STPEngineInitialiser::GLInited = true;
	return true;
}

void STPEngineInitialiser::initCUDA(int device) {
	STPcudaCheckErr(cudaSetDevice(device));
	STPcudaCheckErr(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	STPcudaCheckErr(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
	STPEngineInitialiser::CUDAInited = true;
}

bool STPEngineInitialiser::hasInit() {
	return STPEngineInitialiser::GLInited && 
		STPEngineInitialiser::CUDAInited;
}