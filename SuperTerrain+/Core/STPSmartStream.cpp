#include <SuperTerrain+/Utility/Memory/STPSmartStream.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

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
	this->Stream = unique_ptr<STPStream_t, STPSmartStream::STPStreamDestroyer>(stream);
}

STPSmartStream::STPSmartStream(unsigned int flag, int priority) {
	cudaStream_t stream;
	STPcudaCheckErr(cudaStreamCreateWithPriority(&stream, flag, priority));

	this->Stream = unique_ptr<STPStream_t, STPSmartStream::STPStreamDestroyer>(stream);
}

STPSmartStream::STPStreamPriorityRange STPSmartStream::getStreamPriorityRange() {
	int least, greatest;
	STPcudaCheckErr(cudaDeviceGetStreamPriorityRange(&least, &greatest));

	//we flip the order since CUDA defines the priority as [greatest, least]
	return make_pair(greatest, least);
}

cudaStream_t STPSmartStream::operator*() const noexcept {
	return this->Stream.get();
}