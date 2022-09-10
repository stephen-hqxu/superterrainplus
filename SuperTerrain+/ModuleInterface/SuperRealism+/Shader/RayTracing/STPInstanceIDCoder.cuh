#pragma once
#ifndef _STP_INSTANCEID_CODER_CUH_
#define _STP_INSTANCEID_CODER_CUH_

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPInstanceIDCoder is a simple utility to encode and decode instance ID with extra information.
	*/
	namespace STPInstanceIDCoder {

		//Number of bits available for user ID as defined by OptiX standard.
		//Requires runtime check to make sure user is not using an old version that supports less than this number.
		constexpr static unsigned char UserIDAvailableBit = 28u;
		//The number of bits used by object ID.
		constexpr static unsigned char ObjectIDBit = 14u;
		//The number of bits used by instance ID.
		constexpr static unsigned char InstanceIDBit = UserIDAvailableBit - ObjectIDBit;

		static_assert(ObjectIDBit <= UserIDAvailableBit, "Object ID should not use more bits than the availability.");

		//Mask for valid bits
		constexpr static unsigned int UserIDMask = (1u << UserIDAvailableBit) - 1u,
			InstanceIDMask = (1u << InstanceIDBit) - 1u,
			ObjectIDMask = ((1u << (ObjectIDBit + InstanceIDBit)) - 1u) ^ InstanceIDMask;

		/**
		 * @brief Encode object ID and instance ID into a user ID.
		 * Overflown bits will be set to zero.
		 * @param objectID The object ID
		 * @param instanceID The instance ID.
		 * @return The user ID.
		*/
		__forceinline__ unsigned int encode(unsigned int objectID, unsigned int instanceID) noexcept {
			return (STPInstanceIDCoder::ObjectIDMask & objectID << STPInstanceIDCoder::ObjectIDBit)
				| (instanceID & STPInstanceIDCoder::InstanceIDMask);
		}

		/**
		 * @brief Decode user ID into object ID and instance ID.
		 * @param userID The user ID.
		 * @return The object and instance ID, respectively.
		*/
		__forceinline__ uint2 decode(unsigned int userID) noexcept {
			return make_uint2(
				(userID & STPInstanceIDCoder::ObjectIDMask) >> STPInstanceIDCoder::ObjectIDBit,
				userID & STPInstanceIDCoder::InstanceIDMask
			);
		}
	}

}
#endif//_STP_INSTANCEID_CODER_CUH_