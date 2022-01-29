#version 460 core
#extension GL_ARB_shading_language_include : require

#include </Common/STPAtmosphericScattering.glsl>

layout(early_fragment_tests) in;

//Input
//normalized ray direction, typically a ray cast from the observers eye through a pixel
in vec3 RayDirection;
//Output
layout (location = 0) out vec4 FragColor;

uniform AtmosphereSetting Atmo;
//position of the sun
uniform vec3 SunPosition;

void main(){
	//Compute the resultant scattering factors
	//Normalise sun and view direction
	const AtmosphereComposition comp = atmosphere(Atmo, normalize(SunPosition), normalize(RayDirection));
	
	//Convert the scattering factors into atmoshpere color
	FragColor = vec4(Atmo.iSun * (comp.colSun + comp.colSky), 1.0f);
}