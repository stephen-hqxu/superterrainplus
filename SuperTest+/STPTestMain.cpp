//Catch2
#include <catch2/catch_session.hpp>

//Setup Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
#include <SuperTerrain+/Exception/STPCUDAError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//Test Engine
#include <cuda.h>
#include "STPTestInformation.h"

#include <iostream>

using namespace SuperTerrainPlus;

using std::cerr;
using std::endl;

constexpr static int SelectedDevice = 0;

//Initialise shared data
cudaMemPool_t STPTestInformation::TestDeviceMemoryPool;

int main(int argc, char* argv[]) {
	//setup the engine
	try {
		STPEngineInitialiser::init(SelectedDevice);
	}
	catch (const STPException::STPCUDAError& cuda_err) {
		cerr << cuda_err.what() << endl;
		terminate();
	}
	//get device memory pool
	STPcudaCheckErr(cudaDeviceGetDefaultMemPool(&STPTestInformation::TestDeviceMemoryPool, SelectedDevice));
	cuuint64_t memLimit = 1024ull * 1024ull;//1 MB
	STPcudaCheckErr(cudaMemPoolSetAttribute(STPTestInformation::TestDeviceMemoryPool, cudaMemPoolAttrReleaseThreshold, &memLimit));

	//run catch
	const int result = Catch::Session().run(argc, argv);
	return result;
}