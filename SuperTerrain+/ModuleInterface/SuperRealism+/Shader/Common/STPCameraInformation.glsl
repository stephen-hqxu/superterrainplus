#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	layout(offset = 0) vec3 Position;
	layout(offset = 16) mat4 View;
	layout(offset = 80) mat4 Projection;
	//The values below are calcuated from the values above
	layout(offset = 144) mat4 InvProjection;
	layout(offset = 208) mat4 ProjectionView;
	layout(offset = 272) mat4 InvProjectionView;

	//Depth buffer tweaking
	layout(offset = 336) float LogConstant;
	layout(offset = 340) float Far;
} Camera;

/* -------------------------------------------------------------------- */
#ifdef EMIT_LOG_DEPTH_IMPL
//Convert the clip space position from linear to logarithm depth scale
vec4 linearToLogarithmDepth(vec4 clip) {
	clip.z = log2(Camera.LogConstant * clip.z + 1.0f) / log2(Camera.LogConstant * Camera.Far + 1.0f) * clip.w;
	return clip;
}
#endif//EMIT_LOG_DEPTH_IMPL

/* --------------------------------------------------------------------- */
//Define this macro to use implementation to reconstruct depth to position
//Define to 0 to convert to world space, define to 1 to convert to view space
#ifdef EMIT_DEPTH_RECON_IMPL
#if EMIT_DEPTH_RECON_IMPL == 0
#define DEPTH_CONVERSION_MAT Camera.InvProjectionView
#elif EMIT_DEPTH_RECON_IMPL == 1
#define DEPTH_CONVERSION_MAT Camera.InvProjection
#endif

//Reconstruct fragment world/view position from fragment depth using camera matrix
//depth should have range [0, 1]
//fragment texture coordinate should also be in [0, 1] range
vec3 fragDepthReconstruction(float frag_depth, vec2 frag_coord) {
	//OpenGL requires NDC to be in range [-1, 1], so we need to convert the range
	const vec4 position_ndc = vec4(vec3(frag_coord, frag_depth) * 2.0f - 1.0f, 1.0f),
		position_world = DEPTH_CONVERSION_MAT * position_ndc;

	return position_world.xyz / position_world.w;
}

#undef DEPTH_CONVERSION_MAT
#endif//EMIT_DEPTH_RECON_IMPL

#endif//_STP_CAMERA_INFORMATION_GLSL_