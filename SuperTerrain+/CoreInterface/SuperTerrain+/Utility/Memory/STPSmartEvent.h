#pragma once
#ifndef _STP_SMART_EVENT_H_
#define _STP_SMART_EVENT_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <memory>
#include <type_traits>
//CUDA
#include <cuda_runtime.h>

namespace SuperTerrainPlus {

	class STP_API STPSmartEvent {
	private:

		/**
		 * @brief The default deleter for unique_ptr, which calls the CUDA API to destroy an event
		*/
		struct STP_API STPEventDestroyer {
		public:

			void operator()(cudaEvent_t) const;

		};
		using STPEvent_t = std::remove_pointer_t<cudaEvent_t>;
		//the smart event
		std::unique_ptr<STPEvent_t, STPEventDestroyer> Event;

	public:

		/**
		 * @brief Create a smart CUDA event object.
		 * @param flag Specifies the flag for the created event; default is cudaEventDefault.
		*/
		STPSmartEvent(unsigned int = cudaEventDefault);

		STPSmartEvent(const STPSmartEvent&) = delete;

		STPSmartEvent(STPSmartEvent&&) = default;

		STPSmartEvent& operator=(const STPSmartEvent&) = delete;

		STPSmartEvent& operator=(STPSmartEvent&&) = default;

		~STPSmartEvent() = default;

		/**
		 * @brief Retrieve the underlying managed CUDA event object.
		 * @return The managed CUDA event.
		*/
		cudaEvent_t operator*() const noexcept;

	};

}
#endif//_STP_SMART_EVENT_H_