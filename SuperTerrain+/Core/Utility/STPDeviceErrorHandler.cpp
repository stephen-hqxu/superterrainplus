#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//SQLite
#include <SuperTerrain+/STPSQLite.h>
//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
//System
#include <iostream>
#include <sstream>
#include <typeinfo>

#include <SuperTerrain+/Exception/STPCUDAError.h>
#include <SuperTerrain+/Exception/STPDatabaseError.h>

using std::stringstream;
using std::cerr;
using std::endl;

//always throw an exception
template<class E>
[[noreturn]] inline void printError(stringstream& msg, bool no_msg) noexcept(false) {
	if (!no_msg) {
		cerr << msg.str();
	}
	//throw exception
	throw E(msg.str().c_str());
}

//Helpers to cut down coding efforts
#define ASSERT_FUNCTION(ERR) template<> STP_API void SuperTerrainPlus::STPEngineAssert<ERR>(ERR err_code, const char* __restrict file, const char* __restrict function, int line, bool no_msg) noexcept(false)
#define WRITE_ERR_STRING(SS) SS << file << "(" << function << "):" << line
#define CALL_PROGRAM_DECISION_CUDA(SS) printError<STPException::STPCUDAError>(SS, no_msg)
#define CALL_PROGRAM_DECISION_SQLITE(SS) printError<STPException::STPDatabaseError>(SS, no_msg)

//explicit instantiation
ASSERT_FUNCTION(cudaError_t) {
	//CUDA Runtime API
	if (err_code != cudaSuccess) {
		stringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nCUDA Runtime API: " << cudaGetErrorString(err_code) << endl;
		CALL_PROGRAM_DECISION_CUDA(err_str);
	}
}

ASSERT_FUNCTION(nvrtcResult) {
	//CUDA Runtime Complication
	if (err_code != NVRTC_SUCCESS) {
		stringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nCUDA Runtime Compiler: " << nvrtcGetErrorString(err_code) << endl;
		CALL_PROGRAM_DECISION_CUDA(err_str);
	}
}

ASSERT_FUNCTION(CUresult) {
	//CUDA Driver API
	if (err_code != CUDA_SUCCESS) {
		stringstream err_str;
		const char* str;
		cuGetErrorString(err_code, &str);
		WRITE_ERR_STRING(err_str) << "\nCUDA Driver API: " << str << endl;
		CALL_PROGRAM_DECISION_CUDA(err_str);
	}
}

ASSERT_FUNCTION(int) {
	//SQLite3
	if (err_code != SQLITE_OK) {
		stringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nSQLite: " << sqlite3_errstr(err_code) << endl;
		CALL_PROGRAM_DECISION_SQLITE(err_str);
	}
}