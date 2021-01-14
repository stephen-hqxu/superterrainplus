#version 460 core
#extension GL_ARB_bindless_texture : require

//Input
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec3 Tangent;
layout (location = 4) in vec3 Bitangent;
layout (location = 5) in vec2 BaseChunkPosition;//The top-left coordinate of each chunk in world position, contains x and z

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
	flat unsigned int chunkID;
} vs_out;

uniform mat4 Model;//The model matrix will be used to offset and scale unit planes globally

//auxiliary parameters
uniform uvec2 rendered_chunk_num;//how many chunks will be rendered around the camera?
uniform uvec2 chunk_dimension;//how many unit plane each chunk is consist of?

void main(){
	//We need to instancing the unit plane and build up chunks
	//Each chunk is again build by multiple unit planes
	//So the number of instance will be chunk_size*number_of_chunk
	//Chunks will be built up from top-left to bottom-right corner
	//Unit planes in each chunk will be again built from top-left to bottom-right

	const uint unitplane_per_chunk = chunk_dimension.x * chunk_dimension.y;
	//We need to calculate the starting unit plane, i.e., the unit plane of the top-left corner of the entire rendered chunk
	//const vec2 base_position = Position.xz - (floor(rendered_chunk_num / 2) * chunk_dimension);//We don't need y, all the plane will be aligned at the same height
	//Then find the local unit plane starting position
	//const vec2 chunk_position = vec2(mod(floor(gl_InstanceID / unitplane_per_chunk), rendered_chunk_num.x), 
	//floor(gl_InstanceID / (unitplane_per_chunk * rendered_chunk_num.x)));
	//vec2 local_plane_position = base_position + chunk_position * chunk_dimension;
	//-------------I just realised the codes above are pretty much the same everyframe, so we calculate them on cpu, store them permenately and reuse every frame---------------------
	
	//Also we need to calculate the chunk ID for later shader
	vs_out.chunkID = floatBitsToUint(floor(gl_InstanceID / unitplane_per_chunk));
	//We also calculate the local instance ID, i.e., treating each chunk as an individual, imaging we are not doing any instancing.
	const float localInstanceID = mod(gl_InstanceID, unitplane_per_chunk);
	//BaseChunkPosition is calculate on cpu on init, and reuse every frame, it should be faster
	//remember BaseChunkPosition defines the x and z position
	vec2 local_plane_position = Position.xz + BaseChunkPosition;
	vec2 local_plane_uv = TexCoord;
	
	//Preparation done, starting moving instanced unit planes
	//the x,y offset of the local unit plane within the chunk we are currently in
	vec2 local_offset = vec2(mod(localInstanceID, chunk_dimension.x), floor(localInstanceID / chunk_dimension.y));
	//offset the local unit plane
	local_plane_position += local_offset;
	//remap the texture uv
	local_plane_uv /= chunk_dimension;
	local_plane_uv += vec2(1.0f / chunk_dimension) * local_offset;
	
	//Output
	gl_Position = Model * vec4(local_plane_position.x, Position.y, local_plane_position.y, 1.0f);
	vs_out.texCoord = local_plane_uv;
	vs_out.normal = Normal;
	vs_out.tangent = Tangent;
	vs_out.bitangent = Bitangent;
}