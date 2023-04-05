#pragma once
#ifndef _STP_PLATFORM_H_
#define _STP_PLATFORM_H_

#define STP_PLATFORM_EXPAND_PRAGMA(ARG) _Pragma(#ARG)

/* --------------------------- Compiler Detection ------------------------- */
#define STP_PLATFORM_COMPILER_UNKNOWN 0u
#define STP_PLATFORM_COMPILER_MSVC 1u << 0u /**< Microsoft Visual Compiler */
#define STP_PLATFORM_COMPILER_GCC 1u << 1u /**< GNU C++ Compiler */
#define STP_PLATFORM_COMPILER_CLANG 1u << 2u /**< Clang C++ Compiler */

#define STP_PLATFORM_COMPILER_NVCC 1u << 3u /**< Nvidia CUDA Compiler */
#define STP_PLATFORM_COMPILER_NVRTC 1u << 4u /**< Nvidia CUDA Runtime Compiler */
#define STP_PLATFORM_COMPILER_CUDA (STP_PLATFORM_COMPILER_NVCC | STP_PLATFORM_COMPILER_NVRTC) /**< CUDA Offline/Online Compiler */

//We assume MSVC is the only compiler when running on Windows.
//CUDA enforces that MSVC is the only working compiler on Windows.
//In such case CygWin and MinGW would not work.
#if defined(_MSC_VER)
#define STP_PLATFORM_COMPILER_HOST STP_PLATFORM_COMPILER_MSVC
#elif defined(__GNUC__)
#define STP_PLATFORM_COMPILER_HOST STP_PLATFORM_COMPILER_GCC
#elif defined(__clang__)
#define STP_PLATFORM_COMPILER_HOST STP_PLATFORM_COMPILER_CLANG
#else
#define STP_PLATFORM_COMPILER_HOST STP_PLATFORM_COMPILER_UNKNOWN
#endif

//NVCC is a bit special, it is a compiler driver so needs to be used with a host compiler.
//This means both host compiler flag and NVCC compiler flag will be set.
#if defined(__CUDACC_RTC__)
#define STP_PLATFORM_COMPILER_DEVICE STP_PLATFORM_COMPILER_NVRTC
#elif defined(__CUDACC__)//both CUDA offline and online compiler defines this macro
#define STP_PLATFORM_COMPILER_DEVICE STP_PLATFORM_COMPILER_NVCC
#else
#define STP_PLATFORM_COMPILER_DEVICE STP_PLATFORM_COMPILER_UNKNOWN
#endif

#define STP_PLATFORM_COMPILER (STP_PLATFORM_COMPILER_HOST | STP_PLATFORM_COMPILER_DEVICE)

/************************************************************************************************
* The compiler warning utility provides a compiler independent way to
* help you to tell the compiler to shush.
* 
* - `STP_COMPILER_WARNING_PUSH` saves the current warning setting.
* - `STP_COMPILER_WARNING_SUPPRESS_<COMPILER_NAME>` to suppress specific warning(s).
* - `STP_COMPILER_WARNING_POP` restores the previous warning setting.
* 
* When specifying warning to suppress, the warning(s) is/are specific to compiler,
* so use the correct suppress macro and the warning name as argument as per your compiler.
* Some compiler takes warning as a warning number, whereas others take a string of warning name.
* You may use the suppress function for more than one compiler at a time,
* so it will take different effect when working with different compilers.
************************************************************************************************/
#if STP_PLATFORM_COMPILER & STP_PLATFORM_COMPILER_MSVC
#define STP_COMPILER_WARNING_PUSH					STP_PLATFORM_EXPAND_PRAGMA(warning(push))
#define STP_COMPILER_WARNING_SUPPRESS_MSVC(NAME)	STP_PLATFORM_EXPAND_PRAGMA(warning(disable : NAME))
#define STP_COMPILER_WARNING_POP					STP_PLATFORM_EXPAND_PRAGMA(warning(pop))
#elif STP_PLATFORM_COMPILER & STP_PLATFORM_COMPILER_GCC
#define STP_COMPILER_WARNING_PUSH					STP_PLATFORM_EXPAND_PRAGMA(GCC diagnostic push)
#define STP_COMPILER_WARNING_SUPPRESS_GCC(NAME)		STP_PLATFORM_EXPAND_PRAGMA(GCC diagnostic ignored NAME)
#define STP_COMPILER_WARNING_POP					STP_PLATFORM_EXPAND_PRAGMA(GCC diagnostic pop)
#elif STP_PLATFORM_COMPILER & STP_PLATFORM_COMPILER_CLANG
#define STP_COMPILER_WARNING_PUSH					STP_PLATFORM_EXPAND_PRAGMA(clang diagnostic push)
#define STP_COMPILER_WARNING_SUPPRESS_CLANG(NAME)	STP_PLATFORM_EXPAND_PRAGMA(clang diagnostic ignored NAME)
#define STP_COMPILER_WARNING_POP					STP_PLATFORM_EXPAND_PRAGMA(clang diagnostic pop)
#else
#define STP_COMPILER_WARNING_PUSH
#define STP_COMPILER_WARNING_POP
#endif

//NVCC uses a separate warning system than the host compiler, need to take care of that.
#if (STP_PLATFORM_COMPILER & STP_PLATFORM_COMPILER_CUDA) && defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#define STP_COMPILER_WARNING_PUSH_CUDA				STP_PLATFORM_EXPAND_PRAGMA(nv_diagnostic push)
#define STP_COMPILER_WARNING_SUPPRESS_CUDA(NAME)	STP_PLATFORM_EXPAND_PRAGMA(nv_diag_suppress NAME)
#define STP_COMPILER_WARNING_POP_CUDA				STP_PLATFORM_EXPAND_PRAGMA(nv_diagnostic pop)
#else
#define STP_COMPILER_WARNING_PUSH_CUDA
#define STP_COMPILER_WARNING_POP_CUDA
#endif

//Define suppress macro functions for all compilers, if not defined already.
#ifndef STP_COMPILER_WARNING_SUPPRESS_MSVC
#define STP_COMPILER_WARNING_SUPPRESS_MSVC(NAME)
#endif//MSVC
#ifndef STP_COMPILER_WARNING_SUPPRESS_GCC
#define STP_COMPILER_WARNING_SUPPRESS_GCC(NAME)
#endif//GCC
#ifndef STP_COMPILER_WARNING_SUPPRESS_CLANG
#define STP_COMPILER_WARNING_SUPPRESS_CLANG(NAME)
#endif//CLANG
#ifndef STP_COMPILER_WARNING_SUPPRESS_CUDA
#define STP_COMPILER_WARNING_SUPPRESS_CUDA(NAME)
#endif//CUDA

#endif//_STP_PLATFORM_H_