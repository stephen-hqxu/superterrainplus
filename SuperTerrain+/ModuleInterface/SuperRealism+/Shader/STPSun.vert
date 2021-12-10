#version 460 core

//The position should be a unit cube
layout (location = 0) in vec3 Position;

layout (binding = 0) readonly restrict buffer SkySpaceTransformation{
	mat4 View;
	mat4 Projection;
};

//Output
out vec3 RayDirection;

void main(){
	const mat3 viewRotation = mat3(View);
	const vec3 SkyBoxPosition = viewRotation * vec3(Position);
	RayDirection = normalize(SkyBoxPosition);
	
	//An optimisation, depth buffer of this sky box will always be 1.0
	//So the sky box will only be rendered when there is no visible object in between
	//depth function should be set to less than or equal to
	gl_Position = (Projection * vec4(SkyBoxPosition, 1.0f)).xyww;
}