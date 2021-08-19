//An inlined header for the thread pool template

//DO NOT INCLUDE THIS FILE SEPARARELY, IT'S AUTOMATICALLY MANAGED
#ifdef _STP_THREAD_POOL_H_
template<class F, class ...Args>
std::future<typename std::invoke_result<F, Args...>::type> SuperTerrainPlus::STPThreadPool::enqueue_future(F&& function, Args && ... args) {
	{
		std::shared_lock<std::shared_mutex> lock(this->task_queue_locker);
		//if not running, throw exception
		if (!this->running) {
			throw std::runtime_error("thread pool is not running");
		}
	}
	//get the return type of the function
	using return_type = typename std::invoke_result<F, Args...>::type;
	//create future shared state
	auto new_task = std::make_shared<std::packaged_task<return_type()>>(
		//packaged_task is similar to future but needs to be started explicitly
		std::bind(std::forward<F>(function), std::forward<Args>(args)...));
	std::future<return_type> func_return = new_task->get_future();

	//start sending work
	{
		std::unique_lock<std::shared_mutex> lock(this->task_queue_locker);
		this->task.emplace([new_task]() -> void {
			(*new_task)();
			});
	}

	this->condition.notify_one();
	return func_return;
}

template<class F, class ...Args>
void SuperTerrainPlus::STPThreadPool::enqueue_void(F&& function, Args&& ... args) {
	{
		std::shared_lock<std::shared_mutex> lock(this->task_queue_locker);
		//if not running, throw exception
		if (!this->running) {
			throw std::runtime_error("thread pool is not running");
		}
	}
	auto new_task = std::bind(std::forward<F>(function), std::forward<Args>(args)...);

	//start sending work
	{
		std::unique_lock<std::shared_mutex> lock(this->task_queue_locker);
		this->task.emplace(new_task);
	}

	this->condition.notify_one();
}
#endif