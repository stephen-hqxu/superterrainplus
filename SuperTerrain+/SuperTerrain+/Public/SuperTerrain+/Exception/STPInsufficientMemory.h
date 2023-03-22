#pragma once
#ifndef _STP_INSUFFICIENT_MEMORY_H_
#define _STP_INSUFFICIENT_MEMORY_H_

#include "STPFundamentalException.h"

//assert that there is enough memory to be requested.
#define STP_ASSERTION_MEMORY_SUFFICIENCY(CURR, REQ, MAX, UNIT) \
STP_ASSERTION_EXCEPTION(STPInsufficientMemory, (CURR + REQ <= MAX), CURR, REQ, MAX, UNIT)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPInsufficientMemory is generated when the request cannot be completed due to limited memory.
	*/
	class STP_API STPInsufficientMemory : public STPFundamentalException::STPAssertion {
	public:

		//The current amount, requested amount and the maximum amount of memory available.
		const size_t CurrentMemory, RequestMemory, MaxMemory;
		//The unit of memory.
		const std::string MemoryUnit;

		/**
		 * @param expr The assertion expression.
		 * @param current_memory The current amount of memory consumed.
		 * @param request_memory The amount of memory to be requested.
		 * @param max_memory Tells the maximum available memory. 
		 * @param unit The string representation of the unit of the memory.
		*/
		STPInsufficientMemory(const char*, size_t, size_t, size_t, const char*, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPInsufficientMemory() = default;

	};

}
#endif//_STP_INSUFFICIENT_MEMORY_H_