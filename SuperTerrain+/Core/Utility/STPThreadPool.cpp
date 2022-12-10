#include <SuperTerrain+/Utility/STPThreadPool.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <algorithm>
#include <utility>

using std::thread;
using std::unique_lock;
using std::function;

using std::make_unique;
using std::as_const;

using namespace SuperTerrainPlus;

STPThreadPool::STPThreadPool(const size_t count) : IsPoolRunning(false), IsPoolWaiting(false), PendingTask(0u),
	Worker(make_unique<thread[]>(count)), WorkerCount(count) {
	STP_ASSERTION_NUMERIC_DOMAIN(this->WorkerCount > 0u, "The number of worker in a thread pool must be greater than 0");

	//start the thread pool with forever-spinning threads
	this->IsPoolRunning = true;
	std::generate_n(this->Worker.get(), this->WorkerCount, [this]() {
		return thread([this]() {
			while (true) {
				//next task for execution
				function<void()> task;

				//wait for new task
				unique_lock grab_task_lock(this->TaskQueueLock);
				this->NewTaskNotifier.wait(grab_task_lock,
					[&running = as_const(this->IsPoolRunning), &task_queue = as_const(this->TaskQueue)]() {
						return !(running && task_queue.empty());
					});
				//end the thread if the pool is dead
				if (!this->IsPoolRunning) {
					return;
				}

				//grab the next task
				task = std::move(this->TaskQueue.front());
				this->TaskQueue.pop();

				grab_task_lock.unlock();
				//run the task
				task();
				grab_task_lock.lock();

				this->PendingTask--;
				//notify if someone is waiting for the pool to finish all its tasks
				//lock the mutex while notifying to avoid spurious wake up on the waiting thread
				if (this->IsPoolWaiting) {
					this->TaskDoneNotifier.notify_one();
				}

				//grab_task_lock.unlock();
			}
		});
	});
}

STPThreadPool::~STPThreadPool() {
	//before dying we need to finish everything
	this->waitAll();

	//mark the pool as stopped, and signal to all threads to stop
	this->IsPoolRunning = false;

	this->NewTaskNotifier.notify_all();
	//wait for all of them to finish
	std::for_each_n(this->Worker.get(), this->WorkerCount, [](auto& th) { th.join(); });
}

void STPThreadPool::waitAll() {
	this->IsPoolWaiting = true;

	unique_lock status_check_lock(this->TaskQueueLock);
	this->TaskDoneNotifier.wait(status_check_lock, [&pending_task = as_const(this->PendingTask)]() { return pending_task == 0u; });

	this->IsPoolWaiting = false;
}