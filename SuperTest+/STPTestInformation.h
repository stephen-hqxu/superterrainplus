#pragma once
#ifndef _STP_TEST_INFORMATION_H_
#define _STP_TEST_INFORMATION_H_

#include <cuda_runtime.h>
#include <nanobench.h>

#include <fstream>

/**
 * @brief STPTestInformation contains data to be shared within the test program
*/
namespace STPTestInformation {

	//CUDA device default memory pool, defined in the main file
	extern cudaMemPool_t TestDeviceMemoryPool;

	/**
	 * @brief Render the benchmark result to a file.
	 * @param filename Give a concise name for this benchmark, which will be used as filename.
	 * Please do not include any file extension, the render output is automatically formatted.
	 * @param benchmark The benchmark instance containing the benchmark result.
	*/
	void renderBenchmarkResult(const char*, const ankerl::nanobench::Bench&);

	/**
	 * @brief Create an empty file in the benchmark result directory.
	 * @param filename The filename to be used. No extension is needed.
	 * @return The output file stream created.
	*/
	std::ofstream createBenchmarkResultFile(const char*);

}
#endif//_STP_TEST_INFORMATION_H_