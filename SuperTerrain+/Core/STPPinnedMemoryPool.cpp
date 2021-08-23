#pragma once
#include <Utility/STPPinnedMemoryPool.h>

//Error
#include <SuperError+/STPDeviceErrorHandler.h>
//CUDA
#include <cuda_runtime.h>

using namespace SuperTerrainPlus;

using std::unique_lock;
using std::mutex;

void STPPinnedMemoryPool::STPPinnedMemoryDeleter::operator()(void* ptr) const {
	STPcudaCheckErr(cudaFreeHost(ptr));
}

void STPPinnedMemoryPool::encodeHeader(unsigned char* memory, STPHeader content) {
	STPHeader* headerAddr = reinterpret_cast<STPHeader*>(memory);
	headerAddr[0] = content;
}

STPPinnedMemoryPool::STPHeader STPPinnedMemoryPool::decodeHeader(unsigned char* memory) {
	//header is located in front of the current address, offset by 1 sizeof the header
	STPHeader* headerAddr = reinterpret_cast<STPHeader*>(memory) - 1;
	return headerAddr[0];
}

void* STPPinnedMemoryPool::request(size_t size) {
	unsigned char* memory;

	{
		unique_lock<mutex> lock(this->PoolLock);
		//try to find the memory pool with this size
		STPMemoryPool& pool = this->Collection[size];
		if (pool.empty()) {
			//no memory in it, new allocation
			//need to allocate memory for the header
			STPcudaCheckErr(cudaMallocHost(&memory, size + STPPinnedMemoryPool::HEADER_SIZE));
			//encode the size of the memory as header
			STPPinnedMemoryPool::encodeHeader(memory, size);
		}
		else {
			//we found the pool with that size, release the ownership from the smart pointer
			memory = static_cast<unsigned char*>(pool.front().release());
			pool.pop();
		}
	}

	//return the memory which doesn't contain the header
	return static_cast<void*>(memory + STPPinnedMemoryPool::HEADER_SIZE);
}

void STPPinnedMemoryPool::release(void* memory) {
	unsigned char* raw_memory = static_cast<unsigned char*>(memory);
	//grab the header, which contains the size of the memory
	//currently header only contains the size of the pointer
	const STPHeader size = STPPinnedMemoryPool::decodeHeader(raw_memory);

	{
		unique_lock<mutex> lock(this->PoolLock);
		//put it into the collection
		STPMemoryPool& pool = this->Collection[size];
		//offset the memory so it points to the header
		pool.emplace(static_cast<void*>(raw_memory - STPPinnedMemoryPool::HEADER_SIZE));
	}
}