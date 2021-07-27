#version 460 core

layout (location = 0) in vec2 QuadPosition;
layout (location = 1) in vec2 QuadTexCoord;

out vec2 texcoord_quad;

void main(){
	gl_Position = vec4(QuadPosition, 1.0f, 1.0f);
	texcoord_quad = QuadTexCoord;
}