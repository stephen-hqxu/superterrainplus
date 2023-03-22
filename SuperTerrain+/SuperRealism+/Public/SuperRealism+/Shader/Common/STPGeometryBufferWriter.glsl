#ifndef _STP_GEOMETRY_BUFFER_WRITER_GLSL_
#define _STP_GEOMETRY_BUFFER_WRITER_GLSL_
//A simple wrapper for writing data into geometry buffer for deferred rendering

layout(location = 0) out vec3 gFragAlbedo;
layout(location = 1) out vec3 gFragNormal;
layout(location = 2) out float gFragRoughness;
layout(location = 3) out float gFragAO;
//The following G-Buffers are optional, and will be omitted by GL automatically if there is no attachment
layout(location = 4) out uint gFragMaterial;
//depth is written by fragment shader automatically

//It is better to keep the normal vector normalised, however it will be normalised by GL anyway
//because the pixel format is signed normalised.
void writeGeometryData(const vec3 albedo, const vec3 normal, const float roughness, const float ao, const uint material) {
	gFragAlbedo = albedo;
	gFragNormal = normal;
	gFragRoughness = roughness;
	gFragAO = ao;
	gFragMaterial = material;
}

void writeGeometryData(const vec3 a, const vec3 n, const float r, const float o) {
	writeGeometryData(a, n, r, o, 0u);
}
#endif//_STP_GEOMETRY_BUFFER_WRITER_GLSL_