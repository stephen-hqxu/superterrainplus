#version 460 core

//The position should be a unit cube
layout (location = 0) in vec3 Position;

layout (binding = 0) readonly restrict buffer SkySpaceTransformation{
	mat4 View;
	mat4 ViewRotation;//View matrix with translation removed
	mat4 Projection;
};

//Output
out vec3 RayDirection;

void main(){
	const vec4 SkyBoxPosition = ViewRotation * vec4(Position, 1.0f);
	RayDirection = normalize(SkyBoxPosition.xyz);
	
	//An optimisation, depth buffer of this sky box will always be 1.0
	//So the sky box will only be rendered when there is no visible object in between
	//depth function should be set to less than or equal to
	gl_Position = (Projection * SkyBoxPosition).xyww;
}