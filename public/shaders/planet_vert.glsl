
varying vec2 vUv;
varying vec3 worldNormal;
varying vec3 worldVertPos;


varying vec3 viewNormal;
varying vec3 viewVertPos;

//varying mat4 modelmatrix;


void main(){
	vUv = uv;

	worldNormal  = normalize(modelMatrix * vec4(normal, 0.0)).xyz;
	worldVertPos = vec3(modelMatrix * vec4(position, 1.0 )).xyz;    
		

	vec4 vertPos4 = modelViewMatrix * vec4(position, 1.0);
	viewVertPos = vec3(vertPos4) / vertPos4.w;
	viewNormal =  normalize(vec3(normalMatrix * vec3(normal)));

	//modelmatrix = modelMatrix;

	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

