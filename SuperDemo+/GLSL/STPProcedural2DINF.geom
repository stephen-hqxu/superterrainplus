#version 460 core

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
	vec3 normal;
} gs_out;

//Uniforms
uniform mat4 Model;
uniform vec3 cameraPos;
//The strength of the z component on normalmap
uniform float NormalStrength;
uniform uvec2 HeightfieldDim;

layout (binding = 1) uniform sampler2D Heightfield;

//Functions
void emitFace(int);
//The functions below are used to transformed terrain normal->terrain texture normal, given vertex position, uv and normal
mat2x3 calcTangentBitangent(vec3[3], vec2[3]);
mat3 calcTerrainTBN(mat2x3, vec3);

const ivec2 ConvolutionKernelOffset[8] = {
	{ -1, -1 },
	{  0, -1 },
	{ +1, -1 },
	{ -1,  0 },
	{ +1,  0 },
	{ -1, +1 },
	{  0, +1 },
	{ +1, +1 },
};

void main(){
	//no layer rendering by now, I will add that later
	emitFace(0);
}

void emitFace(int layer){
	/*
		We have three normal systems for terrain, plane normal, terrain normal and terrain texture normal.
		- plane normal is simply (0,1,0), that's how our model is defined (model space normal)
		- terrain normal is the vector that perpendicular to the tessellated terrain
		- terrain texture normal is the normal comes along with the texture on the terrain later in the fragment shader,
		  each texture has its own dedicated normal map

		So we need to do the TBN transform twice: plane->terrain normal then terrain normal->terrain texture normal
	*/
	for(int i = 0; i < gl_in.length; i++){
		//calculate terrain normal from the heightfield
		//the uv increment for each pixel on the heightfield
		const vec2 unit_uv = 1.0f / vec2(HeightfieldDim);

		float cell[ConvolutionKernelOffset.length()];
		//convolve a 3x3 kernel with Sobel operator
		for(int a = 0; a < cell.length(); a++){
			const vec2 uv_offset = unit_uv * ConvolutionKernelOffset[a];
			cell[a] = texture(Heightfield, gs_in[i].texCoord + uv_offset).r;
		}
		//apply filter
		gs_out.normal = normalize(vec3(
			cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7]), 
			cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7]),
			1.0f / NormalStrength
		));
		//transfer the range from [-1,1] to [0,1]
		gs_out.normal = (clamp(gs_out.normal, -1.0f, 1.0f) + 1.0f) / 2.0f;

		//output the primitive information
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