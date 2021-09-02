#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_H_
#define _STP_DEVICE_ERROR_HANDLER_H_

#ifdef STP_ERROR_SEVERITY
#undef STP_ERROR_SEVERITY
#endif

#ifdef STP_CONTINUE_ON_ERROR
#define STP_ERROR_SEVERITY 0u
#elif defined STP_EXCEPTION_ON_ERROR
#define STP_ERROR_SEVERITY 1u
#else
//default is exit
#define STP_ERROR_SEVERITY 2u
#endif

#ifdef STP_SUPPRESS_ERROR_MESSAGE
#undef STP_SUPPRESS_ERROR_MESSAGE
#define STP_SUPPRESS_ERROR_MESSAGE true
#else
#define STP_SUPPRESS_ERROR_MESSAGE false
#endif//STP_SUPPRESS_ERROR_MESSAGE

#include <STPCoreDefine.h>

template<typename Err>
STP_API void STPcudaAssert(Err, unsigned int, const char* __restrict, const char* __restrict, int, bool);

#define STPcudaCheckErr(ans) STPcudaAssert(ans, STP_ERROR_SEVERITY, __FILE__, __FUNCTION__, __LINE__, STP_SUPPRESS_ERROR_MESSAGE)
#endif//_STP_DEVICE_ERROR_HANDLER_H_