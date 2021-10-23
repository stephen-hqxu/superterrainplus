#pragma once
#ifndef _STP_TEST_INFORMATION_H_
#define _STP_TEST_INFORMATION_H_

#include <cuda_runtime.h>

/**
 * @brief STPTestInformation contains data to be shared within the test program
*/
namespace STPTestInformation {

	//CUDA device default memory pool
	extern cudaMemPool_t TestDeviceMemoryPool;

};

#endif//_STP_TEST_INFORMATION_H_