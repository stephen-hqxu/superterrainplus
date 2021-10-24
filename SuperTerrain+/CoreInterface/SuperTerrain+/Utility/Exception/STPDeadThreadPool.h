#pragma once
#ifndef _STP_DEAD_THREAD_POOL_H_
#define _STP_DEAD_THREAD_POOL_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPDeadThreadPool is the error thrown when enqueuing a new task to a thread pool that is not running
	*/
	class STP_API STPDeadThreadPool : public std::runtime_error {
	public:

		/**
		 * @brief Init STPDeadThreadPool
		 * @param msg The message to inform user about the dead thread pool
		*/
		explicit STPDeadThreadPool(const char*);

	};

}
#endif//_STP_DEAD_THREAD_POOL_H_