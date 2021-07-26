#version 460 core

in vec3 TexCoord_vs;

out vec4 FragColor;

uniform float factor;//Determine the factor to blend the day-night cubemap, it's changed in CPU

//The cubemap contains textures for day and night, and we can implement day-night cycle here
layout (binding = 0) uniform samplerCubeArray SkyMap;

void main(){
	FragColor = mix(texture(SkyMap, vec4(TexCoord_vs, 0.0f)), texture(SkyMap, vec4(TexCoord_vs, 1.0f)), factor);

}