#pragma once
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Utility/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>
#include <SuperTerrain+/Utility/Exception/STPDeadThreadPool.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>
#include <SuperTerrain+/Utility/Exception/STPSerialisationError.h>
#include <SuperTerrain+/Utility/Exception/STPUnsupportedFunctionality.h>

using namespace SuperTerrainPlus::STPException;

using std::string;
using std::ios_base;
using std::runtime_error;
using std::invalid_argument;
using std::logic_error;

//STPBadNumericRange.h

STPBadNumericRange::STPBadNumericRange(const char* msg) : STPInvalidArgument(msg) {

}

//STPCompilationError.h

STPCompilationError::STPCompilationError(const char* msg) : STPCUDAError(msg) {

}

//STPCUDAError.h

STPCUDAError::STPCUDAError(const char* msg) : runtime_error(msg) {

}

//STPDeadThreadPool.h

STPDeadThreadPool::STPDeadThreadPool(const char* msg) : runtime_error(msg) {

}

//STPInvalidArgument.h

STPInvalidArgument::STPInvalidArgument(const char* msg) : invalid_argument(msg) {

}

//STPInvalidEnvironment.h

STPInvalidEnvironment::STPInvalidEnvironment(const char* msg) : STPInvalidArgument(msg) {
}

//STPMemoryError.h

STPMemoryError::STPMemoryError(const char* msg) : logic_error(msg) {

}

//STPSerialisationError.h

STPSerialisationError::STPSerialisationError(const char* msg) : failure(msg) {

}

//STPUnsupportedFunctionality.h

STPUnsupportedFunctionality::STPUnsupportedFunctionality(const char* msg) : STPInvalidArgument(msg) {

}