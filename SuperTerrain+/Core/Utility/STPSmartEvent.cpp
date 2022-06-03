#include <SuperTerrain+/Utility/Memory/STPSmartEvent.h>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

using std::unique_ptr;

using namespace SuperTerrainPlus;

void STPSmartEvent::STPEventDestroyer::operator()(cudaEvent_t event) const {
	STPcudaCheckErr(cudaEventDestroy(event));
}

STPSmartEvent::STPSmartEvent(unsigned int flag) {
	cudaEvent_t event;
	STPcudaCheckErr(cudaEventCreateWithFlags(&event, flag));

	this->Event = unique_ptr<STPEvent_t, STPEventDestroyer>(event);
}

cudaEvent_t STPSmartEvent::operator*() const noexcept {
	return this->Event.get();
}