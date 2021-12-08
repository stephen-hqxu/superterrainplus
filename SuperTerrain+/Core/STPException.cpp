#include <SuperTerrain+/Exception/STPAsyncGenerationError.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Exception/STPCUDAError.h>
#include <SuperTerrain+/Exception/STPDatabaseError.h>
#include <SuperTerrain+/Exception/STPDeadThreadPool.h>
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPInvalidSyntax.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
#include <SuperTerrain+/Exception/STPSerialisationError.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

using namespace SuperTerrainPlus::STPException;

using std::string;
using std::ios_base;
using std::runtime_error;
using std::invalid_argument;
using std::logic_error;

//STPAsyncGenerationError.h

STPAsyncGenerationError::STPAsyncGenerationError(const char* msg) : runtime_error(msg) {

}

//STPBadNumericRange.h

STPBadNumericRange::STPBadNumericRange(const char* msg) : STPInvalidArgument(msg) {

}

//STPCompilationError.h

STPCompilationError::STPCompilationError(const char* msg) : STPCUDAError(msg) {

}

//STPCUDAError.h

STPCUDAError::STPCUDAError(const char* msg) : runtime_error(msg) {

}

//STPDatabaseError.h

STPDatabaseError::STPDatabaseError(const char* msg) : runtime_error(msg) {

}

//STPDeadThreadPool.h

STPDeadThreadPool::STPDeadThreadPool(const char* msg) : runtime_error(msg) {

}

//STPGLError/h

STPGLError::STPGLError(const char* msg) : runtime_error(msg) {

}

//STPInvalidArgument.h

STPInvalidArgument::STPInvalidArgument(const char* msg) : invalid_argument(msg) {

}

//STPInvalidEnvironment.h

STPInvalidEnvironment::STPInvalidEnvironment(const char* msg) : STPInvalidArgument(msg) {

}

//STPInvalidSyntax.h

STPInvalidSyntax::STPInvalidSyntax(const char* msg) : runtime_error(msg) {

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