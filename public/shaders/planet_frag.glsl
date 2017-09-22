

#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;

		int shadow;
		float shadowBias;
		float shadowRadius;
		vec2 shadowMapSize;
	};

	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
#endif


#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;

		int shadow;
		float shadowBias;
		float shadowRadius;
		vec2 shadowMapSize;
		float shadowCameraNear;
		float shadowCameraFar;
	};

	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
#endif


varying vec2 vUv;
varying vec3 worldNormal;
varying vec3 worldVertPos;

varying vec3 viewNormal;
varying vec3 viewVertPos;


void main() {

	vec3 worldViewDirection = normalize(cameraPosition - worldVertPos);
	vec3 cameraViewDirection = normalize(viewVertPos); 

	vec4 addedLambert  = vec4(0.0,0.0,0.0, 1.0);

	vec4 addedSpecular = vec4(0.0,0.0,0.0, 1.0);

	#if NUM_DIR_LIGHTS > 0
		for(int i = 0; i < NUM_DIR_LIGHTS; i++) {


			//vec4 lDirection = viewMatrix * vec4( directionalLights[ i ].direction , 0.0 );
			//vec3 lightDir   = normalize( lDirection.xyz );
			vec3 lightDirection   = normalize( directionalLights[ i ].direction );

			//---Diff
			float lambert     = dot( lightDirection , viewNormal );
			addedLambert.rgb += clamp( lambert * directionalLights[ i ].color, 0.0, 1.0);

			addedLambert.rgb *= vec3(0.5,0.5,1.0);

			//---Surface-Spec
			//float specular = clamp(dot(reflect(worldViewDirection, viewNormal), lightDirection) ,0.0, 1.0);
			//float specular = clamp(dot(reflect(cameraViewDirection, viewNormal), lightDirection) ,0.0, 1.0);

			// vec3 L = normalize(lightDirection);
			// vec3 V = normalize(cameraViewDirection);
			// vec3 N = normalize(viewNormal);

			// vec3 H = normalize(L + V);
			// float specular = clamp(dot(N, H),0.0, 1.0);;

			//float specular = clamp(dot(reflect(lightDirection, viewNormal), cameraViewDirection),0.0, 1.0);

			specular = pow(specular, 10.0);

			addedSpecular.rbg += specular;
		}

	#endif
	
	gl_FragColor = addedLambert;
	gl_FragColor += addedSpecular;

}