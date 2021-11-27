#version 460 core
#extension ARB_bindless_texture: require

//Input
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec3 Tangent;
layout (location = 4) in vec3 Bitangent;

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
	vec3 tangent;
	vec3 bitangent;
} vs_out;

uniform mat4 Model;//The model matrix will be used to offset and scale unit planes globally

//auxiliary parameters
uniform uvec2 rendered_chunk_num;//how many chunks will be rendered around the camera?
uniform uvec2 chunk_dimension;//how many unit plane each chunk is consist of?
uniform vec2 base_chunk_position;//The coordinate of the most top-left chunk in world position, contains x and z

void main(){
	//We need to instancing the unit plane and build up chunks
	//Each chunk is again build by multiple unit planes
	//So the number of instance will be chunk_size*number_of_chunk
	//Chunks will be built up from top-left to bottom-right corner
	//Unit planes in each chunk will be again built from top-left to bottom-right

	const uvec2 unitplane_total = rendered_chunk_num * chunk_dimension;
	//remember base_chunk_position defines the x and z position
	vec2 local_plane_position = Position.xz + base_chunk_position;
	vec2 local_plane_uv = TexCoord;
	
	//Preparation done, starting moving instanced unit planes
	//the x,y offset of the local unit plane within the chunk we are currently in
	const vec2 local_offset = vec2(mod(gl_InstanceID, unitplane_total.x), floor(gl_InstanceID / unitplane_total.x));
	//offset the local unit plane
	local_plane_position += local_offset;
	//remap the texture uv
	const vec2 uv_increment = vec2(1.0f / unitplane_total);
	local_plane_uv *= uv_increment;
	local_plane_uv += uv_increment * local_offset;
	
	//Output
	gl_Position = Model * vec4(local_plane_position.x, Position.y, local_plane_position.y, 1.0f);
	vs_out.texCoord = local_plane_uv;
	vs_out.normal = Normal;
	vs_out.tangent = Tangent;
	vs_out.bitangent = Bitangent;
}