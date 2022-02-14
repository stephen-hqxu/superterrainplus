#ifndef _STP_GEOMETRY_BUFFER_WRITER_GLSL_
#define _STP_GEOMETRY_BUFFER_WRITER_GLSL_
//A simple wrapper for writing data into geometry buffer for deferred rendering

layout(location = 0) out vec3 gFragAlbedo;
layout(location = 1) out vec3 gFragNormal;
layout(location = 2) out float gFragSpecular;
layout(location = 3) out float gFragAO;
//depth is writen by fragment shader automatically

//It is better to keep the normal vector normalised, however it will be normalised by GL anyway
//because the pixel format is signed normalised.
void writeGeometryData(vec3 albedo, vec3 normal, float specular, float ao) {
	gFragAlbedo = albedo;
	gFragNormal = normal;
	gFragSpecular = specular;
	gFragAO = ao;
}
#endif//_STP_GEOMETRY_BUFFER_WRITER_GLSL_