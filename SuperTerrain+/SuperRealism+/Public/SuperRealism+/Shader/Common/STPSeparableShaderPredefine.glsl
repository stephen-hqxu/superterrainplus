#ifndef _STP_SEPARABLE_SHADER_PREDEFINE_GLSL_
#define _STP_SEPARABLE_SHADER_PREDEFINE_GLSL_

//This header contains definition for predefined I/O for separable shaders.
//OpenGL requires them to be defined explicitly for separable shaders.

#ifdef SHADER_PREDEFINE_VERT
out gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
#endif//SHADER_PREDEFINE_VERT

#ifdef SHADER_PREDEFINE_TESC
in gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[gl_MaxPatchVertices];

out gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_out[];
#endif//SHADER_PREDEFINE_TESC

#ifdef SHADER_PREDEFINE_TESE
in gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[gl_MaxPatchVertices];

out gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
#endif//SHADER_PREDEFINE_TESE

#ifdef SHADER_PREDEFINE_GEOM
in gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[];

out gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
#endif//SHADER_PREDEFINE_GEOM

#endif//_STP_SEPARABLE_SHADER_PREDEFINE_GLSL_