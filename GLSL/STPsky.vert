#version 460 core

layout (location = 0) in vec3 Position;

out vec3 TexCoord_vs;

layout (std430, binding = 0) buffer PVmatrix{
	layout (offset = 0) mat4 View;
	layout (offset = 64) mat4 View_notrans;//removed translations of view matrix
	layout (offset = 128) mat4 Projection;//each float uses 4 bytes and mat4 has 16 of them
};

uniform mat4 Rotations;

void main(){
	TexCoord_vs = Position;//unit cube, position can be used as texture coord

	gl_Position = vec4(Projection * View_notrans * Rotations * vec4(Position, 1.0f)).xyww;
	//An optimisation, depth buffer of this sky box will always be 1.0
	//So the sky box will only be rendered when there is no visible object in between
	//depth function should be set to less than or equal to
}
