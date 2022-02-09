#version 460 core
//Write additional moment data to the draw buffer
//This moment data is used by variance shadow map and its derivations
#define WRITE_MOMENT 0

#if WRITE_MOMENT
layout(location = 0) out vec2 FragMoment;
#endif

void main(){
#if WRITE_MOMENT
	const float depth = gl_FragCoord.z,
		//moment biasing, this can avoid the shadow acen
		dx = dFdx(depth), dy = dFdy(depth);

	FragMoment = vec2(depth, depth * depth + 0.25f * (dx * dx + dy * dy));
#endif
}