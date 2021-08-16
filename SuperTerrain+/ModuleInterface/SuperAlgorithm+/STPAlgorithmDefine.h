#pragma once
#ifndef _STP_ALGORITHM_DEFINE_H_
#define _STP_ALGORITHM_DEFINE_H_

#if defined(_WIN32) && defined(STPALGORITHMPLUS_HOST_EXPORTS)
#define STPALGORITHMPLUS_HOST_API __declspec(dllexport)
#elif defined(_WIN32)
#define STPALGORITHMPLUS_HOST_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(STPALGORITHMPLUS_HOST_EXPORTS)
#define STPALGORITHMPLUS_HOST_API __attribute__((visibility("default")))
#else
#define STPALGORITHMPLUS_HOST_API
#endif

#endif//_STP_ALGORITHM_DEFINE_H_