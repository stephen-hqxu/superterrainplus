//TEMPLATE DEFINITION FOR SOME THREAD POOL FUNCTIONS
//DO NOT INCLUDE THIS FILE SEPARARELY, IT'S AUTOMATICALLY MANAGED
#ifdef _STP_THREAD_POOL_H_

#include "../Exception/STPNumericDomainError.h"

#include <array>
#include <vector>
#include <variant>

#include <memory>
#include <utility>
#include <exception>

template<class Func, class... Arg, typename Ret>
inline std::future<Ret> SuperTerrainPlus::STPThreadPool::enqueue(Func&& function, Arg&&... args) {
	std::shared_ptr<std::promise<Ret>> new_task_promise = std::make_shared<std::promise<Ret>>();
	this->enqueueDetached([new_task = std::bind(std::forward<Func>(function), std::forward<Arg>(args)...), new_task_promise]() {
		try {
			if constexpr (std::is_void_v<Ret>) {
				std::invoke(new_task);
				new_task_promise->set_value();
			} else {
				new_task_promise->set_value(std::invoke(new_task));
			}
		} catch (...) {
			new_task_promise->set_exception(std::current_exception());
		}
	});

	return new_task_promise->get_future();
}

template<class Func, class... Arg>
inline void SuperTerrainPlus::STPThreadPool::enqueueDetached(Func&& function, Arg&&... args) {
	auto new_task = std::bind(std::forward<Func>(function), std::forward<Arg>(args)...);
	//enqueue new work
	{
		std::unique_lock new_task_lock(this->TaskQueueLock);
		this->TaskQueue.emplace(std::move(new_task));
	}
	//signal a new task
	this->PendingTask++;
	this->NewTaskNotifier.notify_one();
}

template<size_t NB, class Loop, typename IT, typename Ret>
inline auto SuperTerrainPlus::STPThreadPool::enqueueLoop(Loop&& loop, const IT begin_idx, const IT end_idx) {
	static_assert(NB > 1u, "The number of block must be greater than 1 to be parallelised effectively, otherwise please use the non-loop version");
	STP_ASSERTION_NUMERIC_DOMAIN(begin_idx <= end_idx, "the begin index must be less than the end index");
	
	using std::array;
	using std::variant_alternative_t;
	using std::forward;

	//construct return type
	typedef std::future<Ret> STPInvokedFuture;
	//set an arbitrary threshold for using dynamic allocation
	//this is to avoid stack overflow is user splits the task into a huge number of block
	constexpr static bool UseHeapFuture = sizeof(STPInvokedFuture) * NB > 128u;
	typedef std::variant<array<STPInvokedFuture, 1u>,
		std::conditional_t<UseHeapFuture, std::vector<STPInvokedFuture>, array<STPInvokedFuture, NB>>> STPInvokedMultiFuture;

	//calculate begin and end indices for each block
	const size_t total_iteration = static_cast<size_t>(end_idx - begin_idx),
		//if number of iteration is less than block count, step size would be 0; need to take care of that
		step_size = total_iteration / NB;
	if (step_size == 0u) {
		//only one block is needed
		return STPInvokedMultiFuture(variant_alternative_t<0u, STPInvokedMultiFuture> { this->enqueue(forward<Loop>(loop), 0u, begin_idx, end_idx) });
	}

	variant_alternative_t<1u, STPInvokedMultiFuture> blockFuture;
	if constexpr (UseHeapFuture) {
		blockFuture.reserve(NB);
	}
	using std::move;
	for (size_t i = 0u; i < NB; i++) {
		//calculate begin and end for each block
		const IT current_begin = static_cast<IT>(begin_idx + step_size * i),
			//for the last iteration, we want to round up the step size to the total number of iteration
			current_end = static_cast<IT>((i == NB - 1u) ? end_idx : current_begin + step_size);
		STPInvokedFuture current_future = this->enqueue(forward<Loop>(loop), i, current_begin, current_end);

		if constexpr (UseHeapFuture) {
			blockFuture.push_back(move(current_future));
		} else {
			blockFuture[i] = move(current_future);
		}
	}
	return STPInvokedMultiFuture(move(blockFuture));
}
#endif