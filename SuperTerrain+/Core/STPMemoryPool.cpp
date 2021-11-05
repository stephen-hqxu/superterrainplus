#include <SuperTerrain+/Utility/Memory/STPMemoryPool.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>
//CUDA
#include <cuda_runtime.h>

using namespace SuperTerrainPlus;

using std::unique_lock;
using std::mutex;

//Allocate different types of memory
template<STPMemoryPoolType T>
inline static void* allocate(size_t size) {
	if constexpr (T == STPMemoryPoolType::Regular) {
		return malloc(size);
	}
	else {
		void* mem;
		STPcudaCheckErr(cudaMallocHost(&mem, size));
		return mem;
	}
}

template<STPMemoryPoolType T>
void STPMemoryPool<T>::STPMemoryDeleter::operator()(void* ptr) const {
	if constexpr (T == STPMemoryPoolType::Regular) {
		free(ptr);
	}
	else {
		STPcudaCheckErr(cudaFreeHost(ptr));
	}
}

template<STPMemoryPoolType T>
void STPMemoryPool<T>::encodeHeader(unsigned char* memory, STPHeader content) {
	STPHeader* headerAddr = reinterpret_cast<STPHeader*>(memory);
	headerAddr[0] = content;
}

template<STPMemoryPoolType T>
typename STPMemoryPool<T>::STPHeader STPMemoryPool<T>::decodeHeader(unsigned char* memory) {
	//header is located in front of the current address, offset by 1 sizeof the header
	STPHeader* headerAddr = reinterpret_cast<STPHeader*>(memory) - 1;
	return headerAddr[0];
}

template<STPMemoryPoolType T>
void* STPMemoryPool<T>::request(size_t size) {
	if (size == 0ull) {
		throw STPException::STPBadNumericRange("The memory size should be a position integer");
	}

	unsigned char* memory;
	{
		unique_lock<mutex> lock(this->PoolLock);
		//try to find the memory pool with this size
		STPMemoryUnitPool& pool = this->Collection[size];
		if (pool.empty()) {
			//no memory in it, new allocation
			//need to allocate memory for the header
			memory = static_cast<unsigned char*>(allocate<T>(size + STPMemoryPool::HEADER_SIZE));
			//encode the size of the memory as header
			STPMemoryPool::encodeHeader(memory, size);
		}
		else {
			//we found the pool with that size, release the ownership from the smart pointer
			memory = static_cast<unsigned char*>(pool.front().release());
			pool.pop();
		}
	}

	//return the memory which doesn't contain the header
	return static_cast<void*>(memory + STPMemoryPool::HEADER_SIZE);
}

template<STPMemoryPoolType T>
void STPMemoryPool<T>::release(void* memory) {
	unsigned char* raw_memory = static_cast<unsigned char*>(memory);
	//grab the header, which contains the size of the memory
	//currently header only contains the size of the pointer
	const STPHeader size = STPMemoryPool::decodeHeader(raw_memory);

	{
		unique_lock<mutex> lock(this->PoolLock);
		//put it into the collection
		STPMemoryUnitPool& pool = this->Collection[size];
		//offset the memory so it points to the header
		pool.emplace(static_cast<void*>(raw_memory - STPMemoryPool::HEADER_SIZE));
	}
}

//Explicit Instantiations
template class STP_API STPMemoryPool<STPMemoryPoolType::Regular>;
template class STP_API STPMemoryPool<STPMemoryPoolType::Pinned>;