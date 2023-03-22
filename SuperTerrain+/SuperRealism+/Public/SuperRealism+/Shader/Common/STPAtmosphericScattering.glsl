#ifndef _STP_ATMOSPHERIC_SCATTERING_GLSL_
#define _STP_ATMOSPHERIC_SCATTERING_GLSL_

struct AtmosphereSetting {
	float iSun;
	float rPlanet, rAtmos;
	float vAlt;
	vec3 kRlh;
	float kMie, shRlh, shMie, g;
	uint priStep, secStep;
};

struct AtmosphereComposition {
	vec3 colSun, colSky;
};

//ray-sphere intersection that assumes
//the sphere is centred at the origin.
//No intersection when result.x > result.y
vec2 raySphereIntersection(const vec3, const vec3, const float);

//sun position should be a normalised direction to the sun
AtmosphereComposition atmosphere(const AtmosphereSetting atmo, const vec3 sun_pos, const vec3 ray_dir) {
	const vec3 ray_origin = vec3(0.0f, atmo.vAlt, 0.0f);

	//calculate step size of the primary ray
	vec2 p = raySphereIntersection(ray_origin, ray_dir, atmo.rAtmos);
	if (p.x > p.y) {
		//no intersection, default black colour
		return AtmosphereComposition(
			vec3(0.0f),
			vec3(0.0f)
		);
	}
	p.y = min(p.y, raySphereIntersection(ray_origin, ray_dir, atmo.rPlanet).x);
	const float priStepSize = (p.y - p.x) / float(atmo.priStep);

	//Initial primary ray time
	float priTime = 0.0f;

	//Initialise accumulators for Rayleigh and Mie scattering
	vec3 totalRlh = vec3(0.0f), totalMie = vec3(0.0f);

	//Initialise optical depth accumulators for primary ray
	float priOdRlh = 0.0f, priOdMie = 0.0f;

	//Calculate Rayleigh and Mie phases
	const float mu = dot(ray_dir, sun_pos),
		mu_2_p1 = mu * mu + 1.0f,
		g_2 = atmo.g * atmo.g,
		//3.0f / (16.0f * PI)
		pRlh = 0.05968310366 * mu_2_p1,
		//3.0f / (8.0f * PI)
		pMie = 0.1193662073 * (1.0f - g_2) * mu_2_p1 / (pow(1.0f + g_2 - 2.0f * mu * atmo.g, 1.5f) * (2.0f + g_2));

	//Primary ray sampling
	for (uint i = 0u; i < atmo.priStep; i++) {
		//Calculate primary ray sampling position
		const vec3 priPos = ray_origin + ray_dir * (priTime + priStepSize * 0.5f);
		//Calculate sample height
		const float priHeight = length(priPos) - atmo.rPlanet;

		//Calculate optical depth of the Rayleigh and Mie scattering for this step
		const float odStepRlh = exp(-priHeight / atmo.shRlh) * priStepSize,
			odStepMie = exp(-priHeight / atmo.shMie) * priStepSize;
		//Accumulate optical depth
		priOdRlh += odStepRlh;
		priOdMie += odStepMie;

		//Calculate step size of the secondary ray
		const float secStepSize = raySphereIntersection(priPos, sun_pos, atmo.rAtmos).y / float(atmo.secStep);

		//Initialise secondary ray time
		float secTime = 0.0f;

		//Initialise optical depth accumulators for the secondary ray
		float secOdRlh = 0.0f, secOdMie = 0.0f;

		//Secondary ray sampling
		for (uint j = 0u; j < atmo.secStep; j++) {
			//Calculate secondary ray sampling position
			const vec3 secPos = priPos + sun_pos * (secTime + secStepSize * 0.5f);
			//Calculate sample height
			const float secHeight = length(secPos) - atmo.rPlanet;

			//Accumulate optical depth
			secOdRlh += exp(-secHeight / atmo.shRlh) * secStepSize,
				secOdMie += exp(-secHeight / atmo.shMie) * secStepSize;

			//Increment secondary ray time
			secTime += secStepSize;
		}

		//Calculate attenuation
		const vec3 attn = exp(-(atmo.kMie * (priOdMie + secOdMie) + atmo.kRlh * (priOdRlh + secOdRlh)));

		//Accumulate scattering
		totalRlh += odStepRlh * attn;
		totalMie += odStepMie * attn;

		//Increment primary ray time
		priTime += priStepSize;
	}

	//Return the atmosphere composition
	return AtmosphereComposition(
		//Mie scattering emulates sun colour
		pMie * atmo.kMie * totalMie,
		//Whereas Rayleigh scattering emulates sky colour
		pRlh * atmo.kRlh * totalRlh
	);
}

vec2 raySphereIntersection(const vec3 ray_origin, const vec3 ray_direction, const float sphere_radius) {
	const float a = dot(ray_direction, ray_direction),
		b = 2.0f * dot(ray_direction, ray_origin),
		c = dot(ray_origin, ray_origin) - (sphere_radius * sphere_radius),
		det = (b * b) - 4.0f * a * c;

	if (det < 0.0f) {
		return vec2(1e5f, -1e5f);
	}
	//calculate the result of intersection
	const float sqrt_det = sqrt(det);
	vec2 result = vec2(-b);
	result.x -= sqrt_det;
	result.y += sqrt_det;
	return result / (2.0f * a);
}

#endif//_STP_ATMOSPHERIC_SCATTERING_GLSL_