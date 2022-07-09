#include <SuperRealism+/Utility/STPRendererErrorHandler.h>

#define STP_DEVICE_ERROR_HANDLER_BLUEPRINT_IMPLEMENTATION
#include <SuperTerrain+/Utility/STPDeviceErrorHandlerBlueprint.hpp>

//OptiX
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

//well OptiX is based on CUDA, so we just boil it down to CUDA error.
#include <SuperTerrain+/Exception/STPCUDAError.h>

ASSERT_FUNCTION(OptixResult) {
	//OptiX API
	if (err_code != OPTIX_SUCCESS) {
		ostringstream err_str;
		WRITE_ERR_STRING(err_str) << "\nOptiX: " << optixGetErrorString(err_code) << endl;
		printError<STPException::STPCUDAError>(err_str, no_msg);
	}
}