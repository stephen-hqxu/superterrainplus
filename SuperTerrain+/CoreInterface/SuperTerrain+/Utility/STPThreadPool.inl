//TEMPLATE DEFINITION FOR SOME THREAD POOL FUNCTIONS
//DO NOT INCLUDE THIS FILE SEPARARELY, IT'S AUTOMATICALLY MANAGED
#ifdef _STP_THREAD_POOL_H_

#include <memory>
#include <utility>
#include <exception>

template<class Func, class... Args, typename Ret>
inline std::future<Ret> SuperTerrainPlus::STPThreadPool::enqueue(Func&& function, Args&&... args) {
	std::shared_ptr<std::promise<Ret>> new_task_promise = std::make_shared<std::promise<Ret>>();
	this->enqueueDetached([new_task = std::bind(std::forward<Func>(function), std::forward<Args>(args)...), new_task_promise]() {
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

template<class Func, class... Args>
inline void SuperTerrainPlus::STPThreadPool::enqueueDetached(Func&& function, Args&&... args) {
	auto new_task = std::bind(std::forward<Func>(function), std::forward<Args>(args)...);
	//enqueue new work
	{
		std::unique_lock new_task_lock(this->TaskQueueLock);
		this->TaskQueue.emplace(std::move(new_task));
	}
	//signal a new task
	this->PendingTask++;
	this->NewTaskNotifier.notify_one();
}
#endif