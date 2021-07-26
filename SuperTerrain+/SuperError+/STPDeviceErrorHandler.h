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

#if defined (_MSC_VER)
	//MSVC compiler
	#ifdef STPERRORPLUS_EXPORTS
		#define STPERRORPLUS_API __declspec(dllexport)
	#else
		#define STPERRORPLUS_API __declspec(dllimport)
	#endif
#elif defined(__GNUC__)
	//GCC
	#ifdef STPERRORPLUS_EXPORTS
		#define STPERRORPLUS_API __attribute__((visibility("default")))
	#else
		#define STPERRORPLUS_API
	#endif
#else
	//Do nothing for compiler that exports automatically
	#define STPERRORPLUS_API
#endif

template<typename Err>
STPERRORPLUS_API void STPcudaAssert(Err, unsigned int, const char* __restrict, const char* __restrict, int);

#define STPcudaCheckErr(ans) STPcudaAssert(ans, STP_ERROR_SEVERITY, __FILE__, __FUNCTION__, __LINE__);
#endif//_STP_DEVICE_ERROR_HANDLER_H_