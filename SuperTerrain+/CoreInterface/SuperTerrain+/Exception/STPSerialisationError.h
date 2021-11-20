#pragma once
#ifndef _STP_SERIALISATION_ERROR_H_
#define _STP_SERIALISATION_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <ios>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPSerialisationError will be thrown when serialisation fails to operate.
	*/
	class STP_API STPSerialisationError : public std::ios_base::failure {
	public:

		/**
		 * @brief Init STPSerialisationError
		 * @param msg Message for serialisation error
		*/
		explicit STPSerialisationError(const char*);

	};

}
#endif //_STP_SERIALISATION_ERROR_H_