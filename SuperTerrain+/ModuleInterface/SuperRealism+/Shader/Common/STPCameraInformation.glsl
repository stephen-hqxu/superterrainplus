#ifndef _STP_CAMERA_INFORMATION_GLSL_
#define _STP_CAMERA_INFORMATION_GLSL_

/* -------------------------------------- public functions shared externally ----------------------------- */
//OpenGL requires NDC.xy to be in range [-1, 1] and we are using [0, 1] depth
#define STP_DEPTH_BUFFER_TO_NDC(UV, DEPTH) vec4(vec3(UV * 2.0f - 1.0f, DEPTH), 1.0f)

/* ----------------------------------------------- private functions ------------------------------------- */
#ifndef __cplusplus
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
	layout(offset = 448) vec2 LinearDepthFactor;
	layout(offset = 456) float Far;
	layout(offset = 460) float InvFar;//1.0f / Far
} Camera;

/* --------------------------------------------------------------------- */
//Reconstruct fragment world/view position from fragment depth using camera matrix
//depth should have range [0, 1]
//fragment texture coordinate should also be in [0, 1] range
vec3 fragDepthReconstructionGeneric(const mat4 inv_transform, const float frag_depth, const vec2 frag_coord) {
	const vec4 position_scaled = inv_transform * STP_DEPTH_BUFFER_TO_NDC(frag_coord, frag_depth);
	//perform perspective division to un-scale the projection
	return position_scaled.xyz / position_scaled.w;
}

vec3 fragDepthReconstructionWorld(const float frag_depth, const vec2 frag_coord) {
	return fragDepthReconstructionGeneric(Camera.InvProjectionView, frag_depth, frag_coord);
}

vec3 fragDepthReconstructionView(const float frag_depth, const vec2 frag_coord) {
	return fragDepthReconstructionGeneric(Camera.InvProjection, frag_depth, frag_coord);
}

//convert a view position to 2D normalised device coordinate
vec2 fragViewToNDC(const mat4x2 projection_xy, const vec3 position_view) {
	//convert from view to clip space first
	const vec2 position_clip = projection_xy * vec4(position_view, 1.0f);
	//from clip space to NDC by perspective division
	//range convert from [-1, 1] to [0, 1]
	return (position_clip / -position_view.z) * 0.5f + 0.5f;
}

float lineariseDepth(const float depth) {
	//depth remains in the range [0, 1] in reversed, so convert it to [1, 0]
	//both exchanging far and near values and flipping the depth range works
	//here we flip the depth by doing 1 - z
	//far * near / (far - (1 - z) * (far - near))
	return Camera.LinearDepthFactor.x / (Camera.Far - (1.0f - depth) * Camera.LinearDepthFactor.y);
	//linear depth is positive
}
#endif//__cplusplus
#endif//_STP_CAMERA_INFORMATION_GLSL_