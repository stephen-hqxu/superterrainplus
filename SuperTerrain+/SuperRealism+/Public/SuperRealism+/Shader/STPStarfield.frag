#version 460 core
#extension GL_ARB_bindless_texture : require

#define STAR_QUADRATIC_ATTENUATION 1

layout(early_fragment_tests) in;

struct StarfieldSetting{
	float iLklh, OctLklhMul;
	float iScl, OctSclMul;
	float Thres;
	float spdShine;
	float LumMul;
	float MinAlt;
	uint Oct;
};

//Input
in vec3 FragRayDirection;
//Output
layout(location = 0) out vec4 FragColor;

layout(bindless_sampler) uniform sampler1D StarColorSpectrum;

uniform uvec3 RandomSeed;
uniform StarfieldSetting Star;
//control star shining
uniform float ShineTime;

vec3 hash33(const vec3);

void main(){
	const vec3 rayDir = normalize(FragRayDirection);
	if(rayDir.y < Star.MinAlt){
		FragColor = vec4(vec3(0.0f), 1.0f);
		return;
	}

	vec3 starColor = vec3(0.0f);
	float likelihood = Star.iLklh;
	float scale = Star.iScl;
	for(uint i = 0u; i < Star.Oct; i++){
		const vec3 scaled_dir = rayDir * scale,
			//We first divide the sky into a number of grids,
			//we refer one of the four grid vertices closest to the origin as pivot.
			pivot = floor(scaled_dir),
			//A grid contains a number of pixels,
			//and we can locate each pixel within a grid with a local position with respect to the grid.
			//We also want the pivot to be at the centre of the grid instead of on the corner,
			//so we move the grid by half of the grid size towards the origin.
			pixel_grid_local = fract(scaled_dir) - 0.5f;
		
		//now pick up a random number for the grid
		//all pixels in a grid will have the same random number
		const vec3 grid_rand = hash33(pivot);
		//To reduce population of star, only grids that pass the likelihood test host stars.
		if(likelihood >= grid_rand.x){
			//Then add a little bit of animation, we want to make the stars shine.
			//shine intensity is in range [0.0f, 1.0f]
			//add some randomness to avoid having all stars shine at the same frequency
			const float intensity = sin(ShineTime * Star.spdShine * grid_rand.y) * 0.5f + 0.5f,
				//The pivot has the max intensity; as pixels get further away from the grid centre, the intensity drops.
				//So it ends up with a soft edge.
				falloff = 1.0f - smoothstep(0.0f, Star.Thres, length(pixel_grid_local));

			//get some random colours to the star
			starColor += falloff * intensity * textureLod(StarColorSpectrum, grid_rand.z, 0.0f).rgb;
		}
		
		likelihood *= Star.OctLklhMul;
		scale *= Star.OctSclMul;
	}

#if STAR_QUADRATIC_ATTENUATION == 1
	//Apply the optional quadratic attenuation, otherwise attenuation is by default linear
	starColor *= starColor;
#endif

	//Apply luminance multiplier
	FragColor = vec4(starColor * Star.LumMul, 1.0f);
}

vec3 hash33(const vec3 v){
	const vec3 floatMax = 1.0f / vec3(0xFFFFFFFFu);

	//unsigned overflow is OK
	uvec3 p = uvec3(ivec3(v)) + RandomSeed;
	p *= uvec3(374761393u, 1103515245u, 668265263u) + p.zxy + p.yzx;
	p = p.yzx * (p.zxy ^ (p >> 3u));
	return vec3(p ^ (p >> 16u)) * floatMax;
}