#pragma once
#ifndef _STP_FREESLIP_LOCATION_HPP_
#define _STP_FREESLIP_LOCATION_HPP_

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPFreeSlipLocation denotes where tje free-slip data will be available.
	 * Once retrieved, the data retrieved can only be used in designated memory space
	*/
	enum class STPFreeSlipLocation : unsigned char {
		HostMemory = 0x00u,
		DeviceMemory = 0x01u
	};

}
#endif//_STP_FREESLIP_LOCATION_HPP_