#pragma once
#ifndef _STP_THREAD_POOL_H_
#define _STP_THREAD_POOL_H_

#include <STPCoreDefine.h>
//ADT
#include <queue>
//Multi-threading
#include <future>
#include <shared_mutex>
#include <condition_variable>
//System
#include <functional>
#include <type_traits>

#include "Exception/STPDeadThreadPool.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPThreadPool is a simple implementation of thread pool that makes thread constantly working and grabing
	 * new job when possible
	*/
	class STP_API STPThreadPool {
	private:

		//status
		bool running;

		//task queue
		mutable std::shared_mutex task_queue_locker;
		std::queue<std::function<void(void)>> task;
		mutable std::condition_variable_any condition;

		//worker
		std::vector<std::thread> worker;
		
	public:

		/**
		 * @brief Init the thread pool
		 * @param count The number of thread to run constantly in the pool
		*/
		STPThreadPool(unsigned int);

		STPThreadPool(const STPThreadPool&) = delete;

		STPThreadPool(STPThreadPool&&) = delete;

		STPThreadPool& operator=(const STPThreadPool&) = delete;

		STPThreadPool& operator=(STPThreadPool&&) = delete;

		~STPThreadPool();

		/**
		 * @brief Check the number of task that is currently waiting for excecution. Data safety is guaranteed.
		 * @return The number of task that is currently waiting.
		*/
		size_t size() const;

		/**
		 * @brief Check if the thread pool is running. Tasks can only be inserted if the pool is running.
		 * @return True if the pool is running
		*/
		bool isRunning() const;

		/**
		 * @brief Adding new task into the worker queue. Function will continue without waiting for execution.
		 * @tparam F Function type for execution
		 * @tparam ...Args List of types of arguments
		 * @param function Function for execution
		 * @param ...args Lists of arguments
		 * @return The future that holds the return value of the function.
		*/
		template<class F, class ...Args>
		std::future<typename std::invoke_result<F, Args...>::type> enqueue_future(F&&, Args&& ...);

		/**
		 * @brief Adding new task into the worker queue. Function will continue without waiting for execution.
		 * This function will not return future object and the return value of the function is not cared. Task is finished without notification.
		 * @tparam F Function type for execution
		 * @tparam ...Args List of types of arguments
		 * @param function Function for execution
		 * @param ...args Lists of arguments
		*/
		template<class F, class ...Args>
		void enqueue_void(F&&, Args&& ...);

	};
}
#include "STPThreadPool.inl"
#endif//_STP_THREAD_POOL_H_