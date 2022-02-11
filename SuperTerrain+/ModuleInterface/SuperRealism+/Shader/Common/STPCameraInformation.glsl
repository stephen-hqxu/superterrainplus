#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	layout(offset = 0) vec3 Position;
	layout(offset = 16) mat4 View;
	layout(offset = 80) mat4 Projection;
	//The values below are calcuated from the values above
	layout(offset = 144) mat4 ProjectionView;
	layout(offset = 208) mat4 InvProjectionView;

	//Depth buffer tweaking
	layout(offset = 272) float LogConstant;
	layout(offset = 276) float Far;
} Camera;

//Convert the clip space position from linear to logarithm depth scale
vec4 linearToLogarithmDepth(vec4 clip) {
	clip.z = log2(Camera.LogConstant * clip.z + 1.0f) / log2(Camera.LogConstant * Camera.Far + 1.0f) * clip.w;
	return clip;
}

#endif//_STP_CAMERA_INFORMATION_GLSL_