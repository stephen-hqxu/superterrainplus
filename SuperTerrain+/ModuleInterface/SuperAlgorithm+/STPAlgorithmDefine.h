#pragma once
#ifndef _STP_ALGORITHM_DEFINE_H_
#define _STP_ALGORITHM_DEFINE_H_

#if defined(_WIN32) && defined(SUPERALGORITHMPLUS_HOST_EXPORTS)
#define STP_ALGORITHM_HOST_API __declspec(dllexport)
#elif defined(_WIN32)
#define STP_ALGORITHM_HOST_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(SUPERALGORITHMPLUS_HOST_EXPORTS)
#define STP_ALGORITHM_HOST_API __attribute__((visibility("default")))
#else
#define STP_ALGORITHM_HOST_API
#endif

#endif//_STP_ALGORITHM_DEFINE_H_