#pragma once
#ifndef _STP_CORE_DEFINE_H_
#define _STP_CORE_DEFINE_H_

#if defined(_WIN32) && defined(SUPERTERRAINPLUS_EXPORTS)
#define STP_API __declspec(dllexport)
#elif defined(_WIN32)
#define STP_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(SUPERTERRAINPLUS_EXPORTS)
#define STP_API __attribute__((visibility("default")))
#else
#define STP_API
#endif

#endif//_STP_CORE_DEFINE_H_