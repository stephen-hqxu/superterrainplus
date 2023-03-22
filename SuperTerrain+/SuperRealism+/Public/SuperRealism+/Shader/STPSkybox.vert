#version 460 core
#extension GL_ARB_shading_language_include : require

#include </Common/STPCameraInformation.glsl>

//The position should be a unit cube
layout(location = 0) in vec3 Position;

//Output
//A normalized ray direction on the skybox
out vec3 FragRayDirection;

void main(){
	FragRayDirection = normalize(Position);
	
	//For skybox rendering, we want the box to be centred at the origin, i.e., the camera in view space,
	//while it rotates along with the camera.
	//So we remove translation on the view matrix and keep rotation only.

	//An optimisation, depth buffer of this sky box will always be 1.0.
	//So the sky box will only be rendered when there is no visible object in between;
	//Depth function should be set to less than or equal to.
	gl_Position = (Camera.ProjectionViewRotation * vec4(Position, 1.0f)).xyww;
}