#include <Utility/STPThreadPool.h>

using namespace SuperTerrainPlus;

STPThreadPool::STPThreadPool(unsigned int count) {
	if (count == 0u) {
		throw std::invalid_argument("The number of worker in a thread pool must be greater than 0");
	}

	//start the thread pool
	this->running = true;

	//adding non-stopping threads
	this->worker.reserve(static_cast<size_t>(count));
	for (unsigned int i = 0u; i < count; i++) {
		this->worker.emplace_back([this] {
			while (true) {//threads will spin forever
				//next task for execution
				std::function<void()> next_task;

				{
					//get the next task from the queue
					std::unique_lock<std::shared_mutex> lock(this->task_queue_locker);
					//proceeds only if we have new task or the thread has stopped (then we call to exit)
					this->condition.wait(lock, [this] {
						return !(this->running && this->task.empty());
						});
					if (!this->running) {
						//end the thread
						return;
					}
					
					//transfer the function reference then delete from the queue
					next_task = std::move(this->task.front());
					this->task.pop();
				}
				//execution
				next_task();
			}
			
			});
	}
}

STPThreadPool::~STPThreadPool() {
	//mark the pool as stopped, and signal to all threads to stop
	{
		std::unique_lock<std::shared_mutex> lock(this->task_queue_locker);
		this->running = false;
	}

	this->condition.notify_all();
	//join all threads
	for (std::thread& one_worker : this->worker) {
		one_worker.join();
	}
}

size_t STPThreadPool::size() const {
	size_t size;
	{
		std::shared_lock<std::shared_mutex> lock(this->task_queue_locker);
		size = this->task.size();
	}
	return size;
}

bool STPThreadPool::isRunning() const {
	bool isrunning;
	{
		std::shared_lock<std::shared_mutex> lock(this->task_queue_locker);
		isrunning = this->running;
	}
	return isrunning;
}