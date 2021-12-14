#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	layout (offset = 0) vec3 CameraPosition;
	layout (offset = 12) mat4 CameraView;
	layout (offset = 76) mat4 CameraProjection;
};

#endif//_STP_CAMERA_INFORMATION_GLSL_