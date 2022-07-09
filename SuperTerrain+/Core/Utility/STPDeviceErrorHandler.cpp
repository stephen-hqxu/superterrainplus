#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

#define STP_DEVICE_ERROR_HANDLER_BLUEPRINT_IMPLEMENTATION
#include <SuperTerrain+/Utility/STPDeviceErrorHandlerBlueprint.hpp>

//SQLite
#include <SuperTerrain+/STPSQLite.h>
//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <SuperTerrain+/Exception/STPCUDAError.h>
#include <SuperTerrain+/Exception/STPDatabaseError.h>

#define CALL_PROGRAM_DECISION_CUDA(SS) printError<STPException::STPCUDAError>(SS, no_msg)
#define CALL_PROGRAM_DECISION_SQLITE(SS) printError<STPException::STPDatabaseError>(SS, no_msg)

ASSERT_FUNCTION(cudaError_t) {
	//CUDA Runtime API
	if (err_code != cudaSuccess) {
		ostringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nCUDA Runtime API: " << cudaGetErrorString(err_code) << endl;
		CALL_PROGRAM_DECISION_CUDA(err_str);
	}
}

ASSERT_FUNCTION(nvrtcResult) {
	//CUDA Runtime Complication
	if (err_code != NVRTC_SUCCESS) {
		ostringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nCUDA Runtime Compiler: " << nvrtcGetErrorString(err_code) << endl;
		CALL_PROGRAM_DECISION_CUDA(err_str);
	}
}

ASSERT_FUNCTION(CUresult) {
	//CUDA Driver API
	if (err_code != CUDA_SUCCESS) {
		ostringstream err_str;
		const char* str;
		cuGetErrorString(err_code, &str);
		WRITE_ERR_STRING(err_str) << "\nCUDA Driver API: " << str << endl;
		CALL_PROGRAM_DECISION_CUDA(err_str);
	}
}

ASSERT_FUNCTION(int) {
	//SQLite3
	if (err_code != SQLITE_OK) {
		ostringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nSQLite: " << sqlite3_errstr(err_code) << endl;
		CALL_PROGRAM_DECISION_SQLITE(err_str);
	}
}