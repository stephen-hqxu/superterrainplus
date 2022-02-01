#version 460 core
#extension GL_ARB_shading_language_include : require

//The position should be a unit cube
layout (location = 0) in vec3 Position;

#include </Common/STPCameraInformation.glsl>

//Output
out vec3 RayDirection;

void main(){
	RayDirection = normalize(Position);
	
	const mat3 viewRotation = mat3(Camera.View);
	const vec3 SkyBoxPosition = viewRotation * vec3(Position);
	//An optimisation, depth buffer of this sky box will always be 1.0
	//So the sky box will only be rendered when there is no visible object in between
	//depth function should be set to less than or equal to
	gl_Position = (Camera.Projection * vec4(SkyBoxPosition, 1.0f)).xyww;
}