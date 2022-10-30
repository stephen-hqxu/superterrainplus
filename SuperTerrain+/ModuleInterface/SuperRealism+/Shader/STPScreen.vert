#version 460 core
//Input
layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 TexCoord;

//Output
out vec2 FragTexCoord;

void main(){
	FragTexCoord = TexCoord;
	gl_Position = vec4(Position, 0.0f, 1.0f);
}