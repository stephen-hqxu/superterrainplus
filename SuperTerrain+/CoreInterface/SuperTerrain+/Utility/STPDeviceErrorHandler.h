#pragma once
#ifndef _STP_DEVICE_ERROR_HANDLER_H_
#define _STP_DEVICE_ERROR_HANDLER_H_

#include <SuperTerrain+/STPCoreDefine.h>

#define STP_ENGINE_ASSERT_QUAL STP_API
#include "STPDeviceErrorHandlerBlueprint.hpp"

#define STPcudaCheckErr(EC) STP_ASSERT_ENGINE_BASIC(EC)
#define STPsqliteCheckErr(EC) STP_ASSERT_ENGINE_BASIC(EC)

#endif//_STP_DEVICE_ERROR_HANDLER_H_