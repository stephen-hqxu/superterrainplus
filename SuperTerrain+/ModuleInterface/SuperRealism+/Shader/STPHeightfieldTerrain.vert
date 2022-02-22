#version 460 core
//Input
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;

//Output
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
out VertexVS{
	vec2 texCoord;
	vec3 normal;
} vs_out;

uniform mat4 MeshModel;//The model matrix will be used to offset and scale unit planes globally

void main(){
	gl_Position = MeshModel * vec4(Position, 1.0f);
	vs_out.texCoord = TexCoord;
	//our plane is always pointing upwards
	vs_out.normal = vec3(0.0f, 1.0f, 0.0f);
}