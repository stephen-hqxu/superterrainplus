#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_H_
#define _STP_DEVICE_ERROR_HANDLER_H_

#ifdef STP_DEVICE_ERROR_SUPPRESS_CERR
#undef STP_DEVICE_ERROR_SUPPRESS_CERR
#define STP_DEVICE_ERROR_SUPPRESS_CERR true
#else
#define STP_DEVICE_ERROR_SUPPRESS_CERR false
#endif

#include <SuperTerrain+/STPCoreDefine.h>

namespace SuperTerrainPlus {
	template<typename Err>
	STP_API void STPEngineAssert(Err, const char* __restrict, const char* __restrict, int, bool) noexcept(false);
}

#define STPcudaCheckErr(ec) SuperTerrainPlus::STPEngineAssert(ec, __FILE__, __FUNCTION__, __LINE__, STP_DEVICE_ERROR_SUPPRESS_CERR)
#define STPsqliteCheckErr(ec) STPcudaCheckErr(ec)

#endif//_STP_DEVICE_ERROR_HANDLER_H_