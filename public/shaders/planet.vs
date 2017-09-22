<script type="x-shader/x-vertex" id="s">

	/*Three.js added names for vertshader
	34: uniform mat4 modelMatrix;
	35: uniform mat4 modelViewMatrix;
	36: uniform mat4 projectionMatrix;
	37: uniform mat4 viewMatrix;
	38: uniform mat3 normalMatrix;
	39: uniform vec3 cameraPosition;
	40: attribute vec3 position;
	41: attribute vec3 normal;
	42: attribute vec2 uv;
	43: attribute vec2 uv2;
	44: #ifdef USE_COLOR
	45: 	attribute vec3 color;
	46: #endif
	47: #ifdef USE_MORPHTARGETS
	48: 	attribute vec3 morphTarget0;
	49: 	attribute vec3 morphTarget1;
	50: 	attribute vec3 morphTarget2;
	51: 	attribute vec3 morphTarget3;
	52: 	#ifdef USE_MORPHNORMALS
	53: 		attribute vec3 morphNormal0;
	54: 		attribute vec3 morphNormal1;
	55: 		attribute vec3 morphNormal2;
	56: 		attribute vec3 morphNormal3;
	57: 	#else
	58: 		attribute vec3 morphTarget4;
	59: 		attribute vec3 morphTarget5;
	60: 		attribute vec3 morphTarget6;
	61: 		attribute vec3 morphTarget7;
	62: 	#endif
	63: #endif
	64: #ifdef USE_SKINNING
	65: 	attribute vec4 skinIndex;
	66: 	attribute vec4 skinWeight;
	67: #endif
	*/
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


	    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
	}


</script>
