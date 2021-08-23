#pragma once
#include <Utility/STPSmartStream.h>

#include <SuperError+/STPDeviceErrorHandler.h>

using namespace SuperTerrainPlus;

using std::unique_ptr;
using std::pair;
using std::make_pair;

void STPSmartStream::STPStreamDestroyer::operator()(cudaStream_t stream) const {
	STPcudaCheckErr(cudaStreamDestroy(stream));
}

STPSmartStream::STPSmartStream(unsigned int flag) {
	cudaStream_t stream;
	STPcudaCheckErr(cudaStreamCreateWithFlags(&stream, flag));

	//assign the stream
	this->Stream = unique_ptr<CUstream_st, STPSmartStream::STPStreamDestroyer>(stream);
}

STPSmartStream::STPSmartStream(unsigned int flag, int priority) {
	cudaStream_t stream;
	STPcudaCheckErr(cudaStreamCreateWithPriority(&stream, flag, priority));

	this->Stream = unique_ptr<CUstream_st, STPSmartStream::STPStreamDestroyer>(stream);
}

STPSmartStream::STPStreamPriorityRange STPSmartStream::getStreamPriorityRange() {
	int low, high;
	STPcudaCheckErr(cudaDeviceGetStreamPriorityRange(&low, &high));

	return make_pair(low, high);
}

STPSmartStream::operator cudaStream_t() const {
	return this->Stream.get();
}