#pragma once
#ifndef _STP_LOG_STORAGE_HPP_
#define _STP_LOG_STORAGE_HPP_

#include <array>
#include <string>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLogStorage is a log holder for high-level renderers in realism engine.
	*/
	template<size_t L>
	struct STPLogStorage {
	public:

		//The number of log in the storage.
		constexpr static size_t Count = L;

		std::array<std::string, L> Log;

	};

}
#endif//_STP_LOG_STORAGE_HPP_