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

		//The return type after invoking the function.
		template<class Func, class... Arg>
		using STPFunctionReturnType = std::invoke_result_t<std::decay_t<Func>, std::decay_t<Arg>...>;

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
		 * @tparam ...Arg List of types of arguments
		 * @param function Function for execution
		 * @param ...args Lists of arguments
		 * @return The future that holds the return value of the function.
		*/
		template<class Func, class... Arg, typename Ret = STPFunctionReturnType<Func, Arg...>>
		[[nodiscard]] std::future<Ret> enqueue(Func&&, Arg&&...);

		/**
		 * @brief Adding new task into the worker queue. Function will continue without waiting for execution.
		 * This function will not return future object and the return value of the function is discarded.
		 * Task is finished without notification, the application is responsible for checking the task completion state.
		 * @tparam Func Function type for execution
		 * @tparam ...Arg List of types of arguments
		 * @param function Function for execution
		 * @param ...args Lists of arguments
		*/
		template<class Func, class... Arg>
		void enqueueDetached(Func&&, Arg&&...);

		/**
		 * @brief Parallelise a loop by splitting the iterations into equally spaced blocks,
		 * each block is then enqueued to the thread pool as a normal task.
		 * @tparam NB The number of block to use.
		 * Block size is determined automatically based on the number of iteration.
		 * If the number of iteration is less than the number of block used, only one block will be dispatched eventually.
		 * @tparam Loop The type of the loop function.
		 * @tparam IT The type of the index.
		 * @tparam Ret The return type of the loop function.
		 * @param loop The loop function. This function is called once per block.
		 * This function should take 3 argument of type <size_t, IT, IT>: the current block index, the beginning index and the index past the end.
		 * The function should typically contain a `for` loop using the provided begin and end index.
		 * @param begin_idx The beginning index.
		 * @param end_idx The end index, which is 1 past the end.
		 * @return A 2-alternative variant of a static/dynamic array differing in size of future(s).
		 * If the number of iteration is less than number of block specified, the array size is 1.
		 * Otherwise, the size is fixed as the number of block specified by the argument.
		 * If the number of block is very big, or the returned future occupies too much space,
		 * then heap allocation will be used, such that a dynamic array is returned.
		*/
		template<size_t NB, class Loop, typename IT, typename Ret = STPFunctionReturnType<Loop, size_t, IT, IT>>
		[[nodiscard]] auto enqueueLoop(Loop&&, IT, IT);

	};
}
#include "STPThreadPool.inl"
#endif//_STP_THREAD_POOL_H_