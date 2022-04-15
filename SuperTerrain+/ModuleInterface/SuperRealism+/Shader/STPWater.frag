#version 460 core
#extension GL_ARB_shading_language_include : require

//Input
in VertexTES{
	vec3 position_world;
	vec2 texCoord;
} fs_in;

//Output
#include </Common/STPGeometryBufferWriter.glsl>

/* -------------------------- Water Wave Generator ---------------------- */
#include </Common/STPAnimatedWave.glsl>

uniform WaveFunction WaterWave;
//note that this is different from the iteration used in evaluation shader.
uniform unsigned int WaveNormalIteration;
uniform float WaveTime;
/* ---------------------------------------------------------------------- */

#include </Common/STPCameraInformation.glsl>

uniform vec3 Tint;

//compute the surface normal of the water at the current fragment
vec3 calcWaterNormal();

void main(){

}