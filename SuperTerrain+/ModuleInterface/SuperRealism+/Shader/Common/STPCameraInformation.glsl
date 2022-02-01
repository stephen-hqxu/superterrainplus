#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	layout(offset = 0) vec3 Position;
	layout(offset = 16) mat4 View;
	layout(offset = 80) mat4 Projection;
	//The values below are calcuated from the values above
	layout(offset = 144) mat4 ProjectionView;
	layout(offset = 208) mat4 InvProjectionView;
} Camera;

#endif//_STP_CAMERA_INFORMATION_GLSL_