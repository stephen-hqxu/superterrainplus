#ifndef _STP_GEOMETRY_BUFFER_WRITER_GLSL_
#define _STP_GEOMETRY_BUFFER_WRITER_GLSL_
//A simple wrapper for writing data into geometry buffer for deferred rendering

layout(location = 0) out vec3 gFragAlbedo;
layout(location = 1) out vec3 gFragNormal;
layout(location = 2) out float gFragRoughness;
layout(location = 3) out float gFragAO;
//The following G-Buffers are optional, and will be omitted by GL automatically if there is no attachment
layout(location = 4) out unsigned int gFragMaterial;
//depth is written by fragment shader automatically

//It is better to keep the normal vector normalised, however it will be normalised by GL anyway
//because the pixel format is signed normalised.
void writeGeometryData(vec3 albedo, vec3 normal, float roughness, float ao, uint material) {
	gFragAlbedo = albedo;
	gFragNormal = normal;
	gFragRoughness = roughness;
	gFragAO = ao;
	gFragMaterial = material;
}

void writeGeometryData(vec3 a, vec3 n, float r, float o) {
	writeGeometryData(a, n, r, o, 0u);
}
#endif//_STP_GEOMETRY_BUFFER_WRITER_GLSL_