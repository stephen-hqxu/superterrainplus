//Catch2
#include <catch2/catch_session.hpp>

//Test Engine
#include <cuda.h>
#include "STPTestInformation.h"

//Setup Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
#include <SuperTerrain+/Exception/STPCUDAError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>


#include <iostream>

using namespace SuperTerrainPlus;

using std::cerr;
using std::endl;

constexpr static int SelectedDevice = 0;

//Initialise shared data
cudaMemPool_t STPTestInformation::TestDeviceMemoryPool;

int main(const int argc, const char* const* const argv) {
	//setup the engine
	try {
		STPEngineInitialiser::initialise(SelectedDevice, nullptr);
	} catch (const std::exception& err) {
		cerr << err.what() << endl;
		std::terminate();
	}
	//get device memory pool
	STP_CHECK_CUDA(cudaDeviceGetDefaultMemPool(&STPTestInformation::TestDeviceMemoryPool, SelectedDevice));
	cuuint64_t memLimit = 1024ull * 1024ull;//1 MB
	STP_CHECK_CUDA(cudaMemPoolSetAttribute(STPTestInformation::TestDeviceMemoryPool, cudaMemPoolAttrReleaseThreshold, &memLimit));

	//run catch
	const int result = Catch::Session().run(argc, argv);
	return result;
}