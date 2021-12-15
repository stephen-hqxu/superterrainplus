#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	layout (offset = 0) vec3 CameraPosition;
	layout (offset = 16) mat4 CameraView;
	layout (offset = 80) mat4 CameraProjection;
};

#endif//_STP_CAMERA_INFORMATION_GLSL_