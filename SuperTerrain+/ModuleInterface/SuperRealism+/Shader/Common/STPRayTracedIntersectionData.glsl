#ifndef _STP_RAY_TRACED_INTERSECTION_DATA_GLSL_
#define _STP_RAY_TRACED_INTERSECTION_DATA_GLSL_

//record outputs from ray traced intersection shader
//requires bindless texture to convert them to samplers, and the sampler type must match the type as specified in the notes below
layout(std430, binding = 3) readonly restrict buffer STPRayTracedIntersectionData {
	/* --------------- input ----------------- */
	//sampler2D
	layout(offset = 0) uvec2 RayDirection;
	/* --------------- output ---------------- */
	//sampler2D
	layout(offset = 8) uvec2 Position;
	//sampler2D
	layout(offset = 16) uvec2 TexCoord;
	//sampler2D
	layout(offset = 24) uvec2 RayTime;
} RTIntersection;

#endif//_STP_RAY_TRACED_INTERSECTION_DATA_GLSL_