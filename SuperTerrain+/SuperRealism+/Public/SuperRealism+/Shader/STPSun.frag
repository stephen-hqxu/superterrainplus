#version 460 core
#extension GL_ARB_shading_language_include : require

#include </Common/STPAtmosphericScattering.glsl>

layout(early_fragment_tests) in;

//Input
in vec3 FragRayDirection;
//Output
layout(location = 0) out vec4 FragColor;

uniform AtmosphereSetting Atmo;
//position of the sun
uniform vec3 SunPosition;

void main(){
	//Compute the resultant scattering factors
	//Normalise sun and view direction
	const AtmosphereComposition comp = atmosphere(Atmo, normalize(SunPosition), normalize(FragRayDirection));
	
	//Convert the scattering factors into atmosphere colour
	FragColor = vec4(Atmo.iSun * (comp.colSun + comp.colSky), 1.0f);
}