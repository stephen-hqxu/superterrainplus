#version 460 core
#extension GL_ARB_bindless_texture : require

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

//PVM transformation
layout (std430, binding = 0) buffer PVmatrix{
	layout (offset = 0) mat4 View;
	layout (offset = 64) mat4 View_notrans;//removed translations of view matrix
	layout (offset = 128) mat4 Projection;//each float uses 4 bytes and mat4 has 16 of them
};

//Input
in gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[];
in VertexTES{
	vec2 texCoord;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
} gs_in[];
//Output
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
out VertexGS{
	vec4 position_world;
	vec4 position_clip;
	vec2 texCoord;
} gs_out;

//Uniforms
uniform mat4 Model;
uniform vec3 cameraPos;

//Heightfield, RGB is normalmap, A is heightmap
layout (binding = 1) uniform sampler2D Heightfield;

//Functions
void emitFace(int);
//The functions below are used to transformed terrain normal->terrain texture normal, given vertex position, uv and normal
mat2x3 calcTangentBitangent(vec3[3], vec2[3]);
mat3 calcTerrainTBN(mat2x3, vec3);

void main(){
	//no layer rendering by now, I will add that later
	emitFace(0);
}

void emitFace(int layer){
	//calculate tangent and bitangent for each vertices
	const mat2x3 tangent_bitangent = calcTangentBitangent(
	vec3[3](gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz, gl_in[2].gl_Position.xyz),
	vec2[3](gs_in[0].texCoord, gs_in[1].texCoord, gs_in[2].texCoord)
	);

	/*
	We have three normal systems for terrain, plane normal, terrain normal and terrain texture normal.
	- plane normal is simply (0,1,0), that's how our model is defined (model space normal)
	- terrain normal is the vector that perpendicular to the tessellated terrain
	- terrain texture normal is the normal comes along with the texture on the terrain later in the fragment shader,
	  each texture has its own dedicated normal map

	So we need to do the TBN transform twice: plane->terrain normal then terrain normal->terrain texture normal
	*/
	for(int i = 0; i < gl_in.length; i++){
		mat3 TBN_terrain;
		vec3 terrain_normal;
		{
			//The TBN for plane normal->terrain normal is shown below
			const mat3 normalMat = transpose(inverse(mat3(Model)));
			//this function will auto normalised all vectors and re-orthgonalise the tangent specifically
			const mat3 TBN_plane = calcTerrainTBN(//tangent to world
			mat2x3(
			normalMat * gs_in[i].tangent, 
			normalMat * gs_in[i].bitangent), 
			normalMat * gs_in[i].normal);//tangent space to world space

			//We need to calculate TBN terrain normal->terrain texture normal
			//calculate the terrain normal. We need to translate it from [0,1] to [-1,1] first then transform it to world space
			terrain_normal = TBN_plane * (texture(Heightfield, gs_in[i].texCoord).rgb * 2.0f - 1.0f);
			TBN_terrain = transpose(calcTerrainTBN(tangent_bitangent, terrain_normal));//world to tangent
		}

		//output
		gl_Position = gl_in[i].gl_Position;
		gs_out.position_world = gl_Position;
		gl_Position = Projection * View * gl_Position;
		gs_out.position_clip = gl_Position;
		gs_out.texCoord = gs_in[i].texCoord;
		gl_Layer = layer;

		EmitVertex();
	}
	EndPrimitive();
}

mat2x3 calcTangentBitangent(vec3[3] position, vec2[3] uv){
	//edge and delta uv
	const vec3 edge0 = position[1] - position[0], edge1 = position[2] - position[0];
	const vec2 deltauv0 = uv[1] - uv[0], deltauv1 = uv[2] - uv[0];
	//mat(column)x(row). calculate tangent and bitangent
	//since glsl matrix is column major we need to do a lot of transpose
	return transpose(inverse(transpose(mat2(deltauv0, deltauv1))) * transpose(mat2x3(edge0, edge1)));
}

mat3 calcTerrainTBN(mat2x3 tangent_bitangent, vec3 normal){
	//calculate TBN matrix for 3 vertices, tangent space to world space
	const vec3 normal_normalised = normalize(normal), tangent_normalised = normalize(tangent_bitangent[0]);
	return mat3(
		//re-orthgonalise the tangent
		normalize(tangent_normalised - dot(tangent_normalised, normal_normalised) * normal_normalised),
		normalize(tangent_bitangent[1]),
		normal_normalised
	);
}