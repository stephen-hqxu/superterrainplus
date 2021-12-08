#version 460 core

#define PI 3.14159265358979323846264338327923495

struct SkySetting{
	vec3 pSun;
	float iSun;
	float rPlanet, rAtoms;
	vec3 kRlh;
	float kMie, shRlh, shMie, g;
	unsigned int priStep, secStep;
};

//Input
//normalized ray direction, typically a ray cast from the observers eye through a pixel
in vec3 RayDirection;
//Output
out vec4 FragColor;

uniform SkySetting Sky;
//ray origin in meters, typically the position of the viewer's eye
uniform vec3 RayOrigin;

// ray-sphere intersection that assumes
// the sphere is centered at the origin.
// No intersection when result.x > result.y
vec2 raySphereIntersection(vec3, vec3, float);
vec3 atomsphere(vec3, vec3);

void main(){
	//Normalise sun and view direction
	FragColor = vec4(atomsphere(normalize(Sky.pSun), normalize(RayDirection)), 1.0f);
}

vec3 atomsphere(vec3 sun_pos, vec3 ray_dir){
	//calculate step size of the primary ray
	vec2 p = raySphereIntersection(RayOrigin, ray_dir, Sky.rAtoms);
	if(p.x > p.y){
		//no intersection, default black color
		return vec3(0.0f);
	}
	p.y = min(p.y, raySphereIntersection(RayOrigin, ray_dir, Sky.rPlanet).x);
	const float priStepSize = (p.y - p.x) / float(Sky.priStep);

	//Initial primary ray time
	float priTime = 0.0f;

	//Initialise accumulators for Rayleigh and Mie scattering
	vec3 totalRlh = vec3(0.0f), totalMie = vec3(0.0f);

	//Initialise optical depth accumulators for primary ray
	float priOdRlh = 0.0f, priOdMie = 0.0f;

	//Calculate Rayleigh and Mie phases
	const float mu = dot(ray_dir, sun_pos),
		mu_2 = mu * mu,
		g_2 = Sky.g * Sky.g,
		pRlh = 3.0f / (16.0f * PI) * (1.0f * mu_2),
		pMie = 3.0f / (8.0f * PI) * ((1.0f - g_2) * (mu_2 + 1.0f)) / (pow(1.0f + g_2 - 2.0f * mu * Sky.g, 1.5f) * (2.0f + g_2));

	//Primary ray sampling
	for(unsigned int i = 0u; i < Sky.priStep; i++){
		//Calculate primary ray sampling position
		const vec3 priPos = RayOrigin + ray_dir * (priTime + priStepSize * 0.5f);
		//Calculate sample height
		const float priHeight = length(priPos) - Sky.rPlanet;

		//Calculate optical depth of the Rayleigh and Mie scattering for this step
		const float odStepRlh = exp(-priHeight / Sky.shRlh) * priStepSize,
			odStepMie = exp(-priHeight / Sky.shMie) * priStepSize;
		//Accumulate optical depth
		priOdRlh += odStepRlh;
		priOdMie += odStepMie;

		//Calculate step size of the secondary ray
		const float secStepSize = raySphereIntersection(priPos, sun_pos, Sky.rAtoms).y / (1.0f * Sky.secStep);

		//Initialise secondary ray time
		float secTime = 0.0f;

		//Initialise optical depth accumulators for the secondary ray
		float secOdRlh = 0.0f, secOdMie = 0.0f;

		//Secondary ray sampling
		for(unsigned int j = 0u; j < Sky.secStep; j++){
			//Calculate secondary ray sampling position
			const vec3 secPos = priPos + sun_pos * (secTime + secStepSize * 0.5f);
			//Calculate sample height
			const float secHeight = length(secPos) - Sky.rPlanet;

			//Accumulate optical depth
			secOdRlh += exp(-secHeight / Sky.shRlh) * secStepSize,
			secOdMie += exp(-secHeight / Sky.shMie) * secStepSize;

			//Increment secondary ray time
			secTime += secStepSize;
		}

		//Calculate attenuation
		const vec3 attn = exp(-(Sky.kMie * (priOdMie + secOdMie) + Sky.kRlh * (priOdRlh + secOdRlh)));

		//Accumualte scttering
		totalRlh += odStepRlh * attn;
		totalMie += odStepMie * attn;

		//Increment primary ray time
		priTime += priStepSize;
	}

	//Calculate and return the final color
	return Sky.iSun * (pRlh * Sky.kRlh * totalRlh + pMie * Sky.kMie * totalMie);
}

vec2 raySphereIntersection(vec3 ray_origin, vec3 ray_direction, float sphere_radius){
	const float a = dot(ray_direction, ray_direction),
		b = 2.0f * dot(ray_direction, ray_origin),
		c = dot(ray_origin, ray_origin) - (sphere_radius * sphere_radius),
		det = (b * b) - 4.0f * a * c;

	if(det < 0.0f){
		return vec2(1e5f, -1e5f);
	}
	//calculate the result of intersection
	const float sqrt_det = sqrt(det);
	vec2 result = vec2(-b);
	result.x -= sqrt_det;
	result.y += sqrt_det;
	return result / (2.0f * a);
}