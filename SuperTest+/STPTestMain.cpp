#pragma once

//Catch2
#include <catch2/catch_session.hpp>

//Setup Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>

#include <iostream>

using namespace SuperTerrainPlus;

using std::cerr;
using std::endl;

int main(int argc, char* argv[]) {
	//setup the engine
	try {
		STPEngineInitialiser::initCUDA(0);
	}
	catch (const STPException::STPCUDAError& cuda_err) {
		cerr << cuda_err.what() << endl;
		terminate();
	}

	//run catch
	const int result = Catch::Session().run(argc, argv);
	return result;
}