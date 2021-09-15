#pragma once
#ifndef _STP_TEST_INFORMATION_H_
#define _STP_TEST_INFORMATION_H_

#include <cuda_runtime.h>

/**
 * @brief STPTestInformation contains data to be shared within the test program
*/
struct STPTestInformation {
private:

	STPTestInformation() = delete;

	~STPTestInformation() = delete;

public:

	//CUDA device default memory pool
	static cudaMemPool_t TestDeviceMemoryPool;

};

#endif//_STP_TEST_INFORMATION_H_