layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	vec3 CameraPosition;
	mat4 CameraView;
	mat4 CameraProjection;
};