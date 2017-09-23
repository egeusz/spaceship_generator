
uniform sampler2D txtrdiff;
uniform sampler2D txtrspec;
uniform sampler2D txtrclouds;
uniform sampler2D txtrclouddisp;

uniform sampler2D txtrlights;
uniform sampler2D txtrlightsscatter;
uniform sampler2D txtrlightsoffset;

uniform sampler2D txtrglow;
uniform sampler2D txtrglowscatter;


uniform vec3   coloratmo;
uniform vec3   coloratmoscatter1;
uniform vec3   coloratmoscatter2;

uniform vec3   hovercolor;
uniform float  clouddispscale;



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

	//vec3 surfaceColor = vec3(116.0/255.0,101.0/255.0,94.0/255.0);

	//vec3 atmocolor = vec3(0.3,0.7,1.0);
	vec3 atmoScatterColor = vec3(1.0,0.31,0.1);
	vec3 atmoNigthScatterColor = vec3(0.27,0.0,1.0);

	//----------

	vec3 worldViewDirection = normalize(cameraPosition - worldVertPos);
	vec3 cameraViewDirection = normalize(viewVertPos); 

	float faceRatio = clamp( -dot(viewNormal,cameraViewDirection), 0.0, 1.0 );
	float faceRatioInv = 1.0 - faceRatio;



	
	
	vec3 addedSpecAtmo    = vec3(0,0,0);
	vec3 addedSpecCloud   = vec3(0,0,0);
	vec3 addedSpecSurface = vec3(0,0,0);

	vec3 addedLambert = vec3(0,0,0);
	//vec3 addedLambertBack = vec3(0,0,0);

	//vec3 addedLambertGlowMask = vec3(0,0,0);

	float addedLightMask = 0.0;
	float addedSunsetRim = 0.0;

	vec3 addedCrepuscularScatter = vec3(0,0,0);


	#if NUM_DIR_LIGHTS > 0
		for(int i = 0; i < NUM_DIR_LIGHTS; i++) {

			vec3 lightDirection   = normalize( directionalLights[ i ].direction );

			//---Diff
			float lambert     = clamp(dot( lightDirection , viewNormal), 0.0, 1.0);
			float lambertBack  = clamp(dot( -lightDirection , viewNormal), 0.0, 1.0);

			addedLambert += clamp( lambert * directionalLights[ i ].color, 0.0, 1.0);
			
			//addedLambertBack += lambertBack;



			float sunset = (1.0 -lambertBack - lambert);//ring between both front and back lamberts
			float crepuscularscatter = sunset*0.3 + sunset*0.7*faceRatioInv;
			addedCrepuscularScatter += clamp( crepuscularscatter * directionalLights[ i ].color, 0.0, 1.0);

			addedLightMask += 1.0 - lambert;// - pow(sunset,3.0)*0.25;
			
			addedSunsetRim += sunset;
			
			//addedLambertBack += clamp( nightscatter * directionalLights[ i ].color, 0.0, 1.0);



			//addedLambertBase += lambert, 0.0, 1.0);
			//addedLambertBaseRev += clamp( lambertRev, 0.0, 1.0);

			// addedLambert.rgb += clamp( lambert * directionalLights[ i ].color, 0.0, 1.0);

			// addedLambert.rgb *= vec3(0.5,0.5,1.0);

			//---Spec
			float specular = clamp(dot(reflect(lightDirection, viewNormal), cameraViewDirection),0.0, 1.0);

			float specular_atmo = pow(specular, 15.0)*pow(faceRatioInv, 4.0);
			float specular_cloud = pow(specular, 4.0)*pow(faceRatio, 1.5)*0.3;
			float specular_surface = pow(specular, 50.0)*pow(faceRatio, 1.5);

			addedSpecAtmo += clamp( specular_atmo * directionalLights[ i ].color, 0.0, 1.0);
			addedSpecCloud += clamp( specular_cloud * directionalLights[ i ].color, 0.0, 1.0);
			addedSpecSurface += clamp( specular_surface * directionalLights[ i ].color, 0.0, 1.0);
		}
	#endif

	//-----------------
	//Wind

		//---Clouds and Cloud Displacement. 
	vec2 vUv_offset1 = vUv;
	vec2 vUv_offset2 = vUv;

	vec4 cloudDispTexture = texture2D(txtrclouddisp, vUv);
	float clouddispscale2= mod(clouddispscale+0.5,1.0);

	vec2 cloudDisp1 = vec2(clouddispscale*(cloudDispTexture.g-0.5)*0.4, clouddispscale*(cloudDispTexture.r-0.5)*0.4);
	vec2 cloudDisp2 = vec2(clouddispscale2*(cloudDispTexture.g-0.5)*0.4, clouddispscale2*(cloudDispTexture.r-0.5)*0.4);

	vUv_offset1 = mod(vUv_offset1+cloudDisp1*0.2,1.0);
	vUv_offset2 = mod(vUv_offset2+cloudDisp2*0.2,1.0);

	vec4 cloudTexture1 = texture2D(txtrclouds, vUv_offset1);
	vec4 cloudTexture2 = texture2D(txtrclouds, vUv_offset2);

	float cloudMix = clamp(abs(-8.0*clouddispscale+4.0)-1.0, 0.0,1.0);


	vec4 cloudTexture = mix(cloudTexture1, cloudTexture2,cloudMix);

	//-----------------

	float cloudAlpha = cloudTexture.a;
	float cloudAlphaInv = (1.0 - cloudTexture.a);

	//add atmosphere haze based on facing ratio
	vec4 atmoHaze = vec4(coloratmo, 1.0);
	vec4 surfaceColor = mix( texture2D(txtrdiff, vUv), atmoHaze, faceRatioInv);

	//add sunset color to surface
	surfaceColor =  mix( surfaceColor, vec4(coloratmoscatter1, 1.0), pow(addedSunsetRim,7.0) );

	//add sunset color to clouds
	cloudTexture =  mix( cloudTexture, vec4(coloratmoscatter1, 1.0), pow(addedSunsetRim,6.0) );

	//-----------------
	//process lambert
	

	

	surfaceColor = mix( surfaceColor, cloudTexture,  cloudTexture.a );


	vec4 surfaceDiffuse = vec4(addedLambert, 1.0)*surfaceColor;



	//vec3 nightscatter = vec3(1.0,1.0,1.0) - addedLambertBack - addedLambert;
	//vec4 nightscatter = vec4(addedLambert*surfaceColor, 1.0);

	vec4 atmoNightScatter = vec4(addedCrepuscularScatter*coloratmoscatter2*0.2, 1.0);





	//-----------------
	//Specular
	vec4 atmoScatter = vec4(addedSpecAtmo*coloratmoscatter1, 1.0);

	vec4 cloudSpec = vec4(addedSpecCloud, 1.0)*(cloudTexture.a);

	vec4 surfaceSpec = vec4(addedSpecSurface, 1.0)*texture2D(txtrspec, vUv)*cloudAlphaInv;

	//-----------------
	//Glow

	vec4 lights = mix(texture2D(txtrlights, vUv), texture2D(txtrlightsscatter, vUv), cloudAlpha);

	float lightVal = clamp( texture2D(txtrlightsoffset, vUv).r + ((addedLightMask * 2.0)-1.0), 0.0, 1.0 );
	lightVal = (floor(lightVal) + lightVal)*0.5;

	//glowNight *= vec4(1.0 - pow(addedLambert, 8.0), 1.0);
	lights = lights*lightVal;





	
	gl_FragColor = surfaceDiffuse;

	///gl_FragColor = texture2D(txtrclouds, vUv);
	
	//gl_FragColor += vec4(addedSunsetRim,addedSunsetRim,addedSunsetRim,1.0); 

	gl_FragColor += lights;

	gl_FragColor += atmoNightScatter;

	gl_FragColor += surfaceSpec;
	gl_FragColor += cloudSpec;
	gl_FragColor += atmoScatter;

}