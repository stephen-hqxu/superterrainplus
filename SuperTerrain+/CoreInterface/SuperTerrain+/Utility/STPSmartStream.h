#pragma once
#ifndef _STP_SMART_STREAM_H_
#define _STP_SMART_STREAM_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <memory>
#include <type_traits>
//CUDA
#include <cuda_runtime.h>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSmartStream manages CUDA stream smartly using RAII idiom, stream will be destroyed automatically when the object is destructed.
	*/
	class STP_API STPSmartStream {
	private:

		/**
		 * @brief The default deleter for unique_ptr, which calls the CUDA API to destroy a stream
		*/
		struct STPStreamDestroyer {
		public:

			void operator()(cudaStream_t) const;

		};

		using STPStream_t = std::remove_pointer_t<cudaStream_t>;

		//The smart stream deleted by custom deleter
		std::unique_ptr<STPStream_t, STPStreamDestroyer> Stream;

	public:

		//A pair of number contains range of possible priority
		//0: the greatest priority
		//1: the least priority
		typedef std::pair<int, int> STPStreamPriorityRange;

		/**
		 * @brief Create a smart CUDA stream
		 * @param flag The flag passed to CUDA stream creator, default is cudaStreamDefault
		*/
		STPSmartStream(unsigned int = cudaStreamDefault);

		/**
		 * @brief Create a smart CUDA stream, with designate flag and priority
		 * @param flag The flag passed to CUDA stream creator
		 * @param priority The priority of the stream
		*/
		STPSmartStream(unsigned int, int);

		~STPSmartStream() = default;

		STPSmartStream(const STPSmartStream&) = delete;

		STPSmartStream(STPSmartStream&&) = default;

		STPSmartStream& operator=(const STPSmartStream&) = delete;

		STPSmartStream& operator=(STPSmartStream&&) = default;

		/**
		 * @brief Get the meaningful value for CUDA stream priority.
		 * @return The valid priority range
		*/
		static STPStreamPriorityRange getStreamPriorityRange();

		/**
		 * @brief Obtained the underlying stream
		*/
		operator cudaStream_t() const;

	};

}
#endif//_STP_SMART_STREAM_H_