//Catch2
#include <catch2/catch_session.hpp>

//Test Engine
#include <cuda.h>
#include <cuda_runtime.h>
#include "STPTestInformation.h"

//Setup Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//IO
#include <iostream>
#include <fstream>
#include <filesystem>

#include <string>

using std::cout, std::cerr, std::endl;
using std::ofstream;

constexpr static int SelectedDevice = 0;
//The root path where the benchmark result should be stored.
constexpr static const char* BenchmarkResultPath = "./BenchmarkResult";

//Initialise shared data
cudaMemPool_t STPTestInformation::TestDeviceMemoryPool;

int main(const int argc, const char* const* const argv) {
	//setup the engine
	try {
		SuperTerrainPlus::STPEngineInitialiser::initialise(SelectedDevice, nullptr);
	} catch (const std::exception& err) {
		cerr << err.what() << endl;
		std::terminate();
	}
	//get device memory pool
	STP_CHECK_CUDA(cudaDeviceGetDefaultMemPool(&STPTestInformation::TestDeviceMemoryPool, SelectedDevice));
	cuuint64_t memLimit = 1024ull * 1024ull;//1 MB
	STP_CHECK_CUDA(cudaMemPoolSetAttribute(STPTestInformation::TestDeviceMemoryPool, cudaMemPoolAttrReleaseThreshold, &memLimit));

	//setup benchmark result directory
	const std::filesystem::path benchmarkPath = BenchmarkResultPath;
	if (std::filesystem::create_directory(benchmarkPath)) {
		cout << "Benchmark result directory \'" << benchmarkPath.relative_path() << "\' does not exist, created a new one" << endl;
	}

	//run catch
	const int result = Catch::Session().run(argc, argv);
	return result;
}

/**
 * @brief Open a result file in the benchmark result directory.
 * @param filename The filename to be opened.
 * @param extension The extension of the file.
 * @return The opened file output stream.
*/
static ofstream openResultFile(const char* const filename, const char* const extension) {
	using std::ios;
	return ofstream(std::string(BenchmarkResultPath) + '/' + filename + extension, ios::out | ios::trunc);
}

void STPTestInformation::renderBenchmarkResult(const char* const filename, const ankerl::nanobench::Bench& benchmark) {
	ofstream rendered = openResultFile(filename, ".html");
	ankerl::nanobench::render(ankerl::nanobench::templates::htmlBoxplot(), benchmark, rendered);
}

ofstream STPTestInformation::createBenchmarkResultFile(const char* const filename) {
	return openResultFile(filename, ".md");
}