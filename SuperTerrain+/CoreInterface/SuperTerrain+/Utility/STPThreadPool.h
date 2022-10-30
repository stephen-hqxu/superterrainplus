#pragma once
#ifndef _STP_THREAD_POOL_H_
#define _STP_THREAD_POOL_H_

#include <SuperTerrain+/STPCoreDefine.h>
//ADT
#include <queue>
//Multi-threading
#include <atomic>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
//System
#include <functional>
#include <type_traits>

namespace SuperTerrainPlus {

	/**
	 * @brief STPThreadPool is a simple implementation of thread pool that makes thread constantly working and grabbing
	 * new job when possible
	*/
	class STP_API STPThreadPool {
	private:

		std::atomic<bool> IsPoolRunning, IsPoolWaiting;
		//increment when new task has arrived, decrement when it is done by a thread
		std::atomic<size_t> PendingTask;

		mutable std::mutex TaskQueueLock;
		mutable std::condition_variable NewTaskNotifier, TaskDoneNotifier;

		std::queue<std::function<void()>> TaskQueue;

		const std::unique_ptr<std::thread[]> Worker;
		
	public:

		const size_t WorkerCount;

		/**
		 * @brief Initialise the thread pool.
		 * @param count The number of thread to run constantly in the pool
		*/
		STPThreadPool(size_t);

		STPThreadPool(const STPThreadPool&) = delete;

		STPThreadPool(STPThreadPool&&) = delete;

		STPThreadPool& operator=(const STPThreadPool&) = delete;

		STPThreadPool& operator=(STPThreadPool&&) = delete;

		~STPThreadPool();

		/**
		 * @brief Wait for all tasks to finish.
		 * This function will return only when all tasks in the task queue are finished, and all threads go back to idle state.
		*/
		void waitAll();

		/**
		 * @brief Adding new task into the worker queue. Function will return immediately without waiting for execution.
		 * @tparam Func Function type for execution
		 * @tparam ...Args List of types of arguments
		 * @param function Function for execution
		 * @param ...args Lists of arguments
		 * @return The future that holds the return value of the function.
		*/
		template<class Func, class... Args, typename Ret = std::invoke_result_t<Func, Args...>>
		[[nodiscard]] std::future<Ret> enqueue(Func&&, Args&&...);

		/**
		 * @brief Adding new task into the worker queue. Function will continue without waiting for execution.
		 * This function will not return future object and the return value of the function is discarded.
		 * Task is finished without notification, the application is responsible for checking the task completion state.
		 * @tparam Func Function type for execution
		 * @tparam ...Args List of types of arguments
		 * @param function Function for execution
		 * @param ...args Lists of arguments
		*/
		template<class Func, class... Args>
		void enqueueDetached(Func&&, Args&&...);

	};
}
#include "STPThreadPool.inl"
#endif//_STP_THREAD_POOL_H_