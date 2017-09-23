
varying vec2 vUv;
varying vec3 worldNormal;
varying vec3 worldVertPos;


varying vec3 viewNormal;
varying vec3 viewVertPos;


void main(){
	vUv = uv;

	worldNormal  = normalize(modelMatrix * vec4(normal, 0.0)).xyz;
	worldVertPos = vec3(modelMatrix * vec4(position, 1.0 )).xyz;    
		

	vec4 vertPos4 = modelViewMatrix * vec4(position, 1.0);
	viewVertPos = vec3(vertPos4) / vertPos4.w;
	viewNormal =  normalize(vec3(normalMatrix * vec3(normal)));

	/*
	viewUV = 


	viewU = vec3(normalMatrix * vUv);
	viewV = vec3(normalMatrix * uv.y);

	//normal_cameraSpace = normalize(modelViewMatrix * vec4(normal, 1.0)).xyz;

	// vec3 objectNormal = vec3( normal );
	// vec3 transformedNormal = normalMatrix * objectNormal;
	// vNormal = normalize( transformedNormal );
	*/

	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

