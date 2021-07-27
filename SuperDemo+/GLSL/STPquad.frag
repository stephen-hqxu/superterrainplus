#version 460 core

in vec2 texcoord_quad;

out vec4 QuadContent;

layout (binding = 0) uniform sampler2D screen;

void main(){
	QuadContent = texture(screen, texcoord_quad);
}