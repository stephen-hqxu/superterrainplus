#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

layout(std430, binding = 0) readonly restrict buffer STPCameraInformation {
	layout(offset = 0) vec3 Position;
	layout(offset = 16) mat4 View;
	layout(offset = 80) mat3 ViewNormal;//normal matrix that encapsulated view matrix only

	layout(offset = 128) mat4 Projection;
	layout(offset = 192) mat4 InvProjection;

	layout(offset = 256) mat4 ProjectionViewRotation;//projection * mat4(mat3(view)), view only keeps rotation part.
	layout(offset = 320) mat4 ProjectionView;
	layout(offset = 384) mat4 InvProjectionView;

	//Camera properties
	layout(offset = 448) vec3 LinearDepthFactor;
	layout(offset = 460) float Far;
	//Of course we can check type of projection by examining the projection matrix, 
	//for example projection is orthographic if and only if Projection[3][3] == 1.0f.
	//It is faster to compute the result on host than computing every frame.
	layout(offset = 464) bool useOrtho;
} Camera;

/* --------------------------------------------------------------------- */
//Define this macro to use implementation to reconstruct depth to position
#if defined(EMIT_DEPTH_RECON_WORLD_IMPL) || defined(EMIT_DEPTH_RECON_VIEW_IMPL)
#ifdef EMIT_DEPTH_RECON_WORLD_IMPL
#define DEPTH_CONVERSION_MAT Camera.InvProjectionView
#elif defined(EMIT_DEPTH_RECON_VIEW_IMPL)
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
#endif//EMIT_DEPTH_RECON_WORLD_IMPL || EMIT_DEPTH_RECON_VIEW_IMPL

#ifdef EMIT_VIEW_TO_NDC_IMPL
//convert a view position to 2D normalised device coordinate
vec2 fragViewToNDC(mat4x2 projection_xy, vec3 position_view) {
	//convert from view to clip space first
	const vec2 position_clip = projection_xy * vec4(position_view, 1.0f);
	//from clip space to NDC by perspective division
	//range convert from [-1, 1] to [0, 1]
	return (position_clip / (Camera.useOrtho ? 1.0f : -position_view.z)) * 0.5f + 0.5f;
}
#endif//EMIT_VIEW_TO_NDC_IMPL

#ifdef EMIT_LINEARISE_DEPTH_IMPL
float lineariseDepth(float depth) {
	//depth needs to be converted to range [-1, 1]
	//2 * far * near / (far + near - (2 * z - 1) * (far - near))
	return Camera.LinearDepthFactor.x / (Camera.LinearDepthFactor.y - (2.0f * depth - 1.0f) * Camera.LinearDepthFactor.z);
}
#endif//EMIT_LINEARISE_DEPTH_IMPL

#endif//_STP_CAMERA_INFORMATION_GLSL_