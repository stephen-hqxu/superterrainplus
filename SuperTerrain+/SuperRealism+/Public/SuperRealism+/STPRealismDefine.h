#pragma once
#ifndef _STP_REALISM_DEFINE_H_
#define _STP_REALISM_DEFINE_H_

#if defined(_WIN32) && defined(SUPERREALISMPLUS_EXPORTS)
#define STP_REALISM_API __declspec(dllexport)
#elif defined(_WIN32)
#define STP_REALISM_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(SUPERREALISMPLUS_EXPORTS)
#define STP_REALISM_API __attribute__((visibility("default")))
#else
#define STP_REALISM_API
#endif

#endif//_STP_REALISM_DEFINE_H_