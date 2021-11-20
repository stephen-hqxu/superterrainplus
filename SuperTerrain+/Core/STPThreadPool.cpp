#include <SuperTerrain+/Utility/STPThreadPool.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>

using namespace SuperTerrainPlus;

STPThreadPool::STPThreadPool(unsigned int count) {
	if (count == 0u) {
		throw STPException::STPBadNumericRange("The number of worker in a thread pool must be greater than 0");
	}

	//start the thread pool
	this->running = true;

	//adding non-stopping threads
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
						//predicate is checked outside the lock, so this is thread safe!
						return !(this->running && this->task.empty());
						});
					if (!this->running && this->task.empty()) {
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
	std::shared_lock<std::shared_mutex> lock(this->task_queue_locker);
	return this->task.size();
}

bool STPThreadPool::isRunning() const {
	std::shared_lock<std::shared_mutex> lock(this->task_queue_locker);
	return this->running;
}