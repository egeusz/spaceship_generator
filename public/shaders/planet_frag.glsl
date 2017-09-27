
#define M_PI 3.1415926535897932384626433832795

#extension GL_OES_standard_derivatives : enable

const highp float;

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
uniform float  clouddisp;

uniform sampler2D txtrbump;
uniform float bumpScale;

// Derivative maps - bump mapping unparametrized surfaces by Morten Mikkelsen
// http://mmikkelsen3d.blogspot.sk/2011/07/derivative-maps.html

// Evaluate the derivative of the height w.r.t. screen-space using forward differencing (listing 2)



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


vec2 dHdxy_fwd() {

	vec2 dSTdx = dFdx( vUv );
	vec2 dSTdy = dFdy( vUv );

	float Hll = bumpScale * texture2D( txtrbump, vUv ).x;
	float dBx = bumpScale * texture2D( txtrbump, vUv + dSTdx ).x - Hll;
	float dBy = bumpScale * texture2D( txtrbump, vUv + dSTdy ).x - Hll;

	return vec2( dBx, dBy );

}

vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy ) {

	// Workaround for Adreno 3XX dFd*( vec3 ) bug. See #9988

	vec3 vSigmaX = vec3( dFdx( surf_pos.x ), dFdx( surf_pos.y ), dFdx( surf_pos.z ) );
	vec3 vSigmaY = vec3( dFdy( surf_pos.x ), dFdy( surf_pos.y ), dFdy( surf_pos.z ) );
	vec3 vN = surf_norm;		// normalized

	vec3 R1 = cross( vSigmaY, vN );
	vec3 R2 = cross( vN, vSigmaX );

	float fDet = dot( vSigmaX, R1 );

	vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
	return normalize( abs( fDet ) * surf_norm - vGrad );

}



void main() {

	float cloudHeight = 20.0;


	//----------
	vec3 viewNormalBump = viewNormal;
	viewNormalBump = perturbNormalArb( viewVertPos, viewNormalBump, dHdxy_fwd() );

	vec3 worldViewDirection = normalize(cameraPosition - worldVertPos);
	vec3 cameraViewDirection = normalize(viewVertPos); 

	float faceRatio = clamp( -dot(viewNormal,cameraViewDirection), 0.0, 1.0 );
	float faceRatioInv = 1.0 - faceRatio;

	//-----------------
	//---Clouds and Cloud Displacement. 

	float offset = sin( clouddisp * 2.0 * M_PI )*0.2;

	vec4 cloudDispTexture = texture2D(txtrclouddisp, vUv);

	vec2 vUv_offset = vUv + vec2(offset*(cloudDispTexture.g-0.5), offset*(cloudDispTexture.r-0.5));
	vUv_offset = mod(vUv_offset,1.0);

	vec4 cloudTexture = texture2D(txtrclouds, vUv_offset);
	

	float cloudAlpha = cloudTexture.a;
	float cloudAlphaInv = (1.0 - cloudTexture.a);

	//-----------------

	/*
	vec3 normal = normalize( viewNormal );


	vec3 eye_pos = -viewVertPos;

	vec3 q0 = vec3( dFdx( eye_pos.x ), dFdx( eye_pos.y ), dFdx( eye_pos.z ) );
	vec3 q1 = vec3( dFdy( eye_pos.x ), dFdy( eye_pos.y ), dFdy( eye_pos.z ) );
	vec2 st0 = dFdx( vUv );
	vec2 st1 = dFdy( vUv );

	vec3 S = normalize( q0 * st1.t - q1 * st0.t );
	vec3 T = normalize( -q0 * st1.s + q1 * st0.s );
	vec3 N = normalize( normal );

	mat3 tsn = mat3( S, T, N );

	vec3 bitanU = tsn * vec3(1.0,0.0,0.0);
	vec3 bitanV = tsn * vec3(0.0,1.0,0.0);
	*/

	//------------------
	
	vec3 addedSpecAtmo    = vec3(0,0,0);
	vec3 addedSpecCloud   = vec3(0,0,0);
	vec3 addedSpecSurface = vec3(0,0,0);

	vec3 addedLambert = vec3(0,0,0);
	vec3 addedLambertBump = vec3(0,0,0);
	//vec3 addedLambertBack = vec3(0,0,0);

	//vec3 addedLambertGlowMask = vec3(0,0,0);

	float addedLightMask = 0.0;
	float addedSunsetRim = 0.0;



	vec3 addedCrepuscularScatter = vec3(0,0,0);

	float addedcloudShadow = 0.0;
	vec2 dSTdx = dFdx( vUv );
	vec2 dSTdy = dFdy( vUv );
	vec3 uvX = vec3(dSTdx, 0.0);
	vec3 uvY = vec3(dSTdy, 0.0);
	
	#if NUM_DIR_LIGHTS > 0
		for(int i = 0; i < NUM_DIR_LIGHTS; i++) {

			vec3 lightDirection   = normalize( directionalLights[ i ].direction );

			//---Diff
			float lambert     = clamp(dot( lightDirection , viewNormal), 0.0, 1.0);
			float lambertBump = clamp(dot( lightDirection , viewNormalBump), 0.0, 1.0);
			float lambertBack  = clamp(dot( -lightDirection , viewNormal), 0.0, 1.0);

			addedLambert += clamp( lambert * directionalLights[ i ].color, 0.0, 1.0);
			addedLambertBump += clamp( lambertBump * directionalLights[ i ].color, 0.0, 1.0);
			//addedLambertBack += lambertBack;



			float sunset = (1.0 -lambertBack - lambert);//ring between both front and back lamberts
			float crepuscularscatter = sunset*0.3 + sunset*0.7*faceRatioInv;
			addedCrepuscularScatter += clamp( crepuscularscatter * directionalLights[ i ].color, 0.0, 1.0);

			addedLightMask += 1.0 - lambert;// - pow(sunset,3.0)*0.25;
			
			addedSunsetRim += sunset;
			
			//---Cloud Shadow
			float shadowOffsetU = dot( lightDirection, uvX)*cloudHeight;
			float shadowOffsetV = dot( lightDirection, uvY)*cloudHeight;

			vec2 vUv_shadow_offset = vec2(vUv_offset.s - shadowOffsetU, vUv_offset.t - shadowOffsetV);
			vUv_shadow_offset = mod(vUv_shadow_offset,1.0);

			addedcloudShadow +=  (1.0 - texture2D( txtrclouds, vUv_shadow_offset ).a);

			//---Spec
			float specular = clamp(dot(reflect(lightDirection, viewNormalBump), cameraViewDirection),0.0, 1.0);

			float specular_atmo = pow(specular, 10.0)*pow(faceRatioInv, 4.0);
			float specular_cloud = pow(specular, 4.0)*pow(faceRatio, 1.5)*0.3;
			float specular_surface = pow(specular, 15.0)*pow(faceRatio, 1.5);

			addedSpecAtmo += clamp( specular_atmo * directionalLights[ i ].color, 0.0, 1.0);
			addedSpecCloud += clamp( specular_cloud * directionalLights[ i ].color, 0.0, 1.0);
			addedSpecSurface += clamp( specular_surface * directionalLights[ i ].color, 0.0, 1.0);
		}
	#endif



	//-----------------

	//add atmosphere haze based on facing ratio
	//vec3 atmoHaze = vec4(coloratmo);
	
	//vec4 surfaceTexture = mix( texture2D(txtrdiff, vUv), atmoHaze, faceRatioInv);

	vec4 surfaceTexture = texture2D(txtrdiff, vUv);

	//add sunset color to surface
	surfaceTexture =  mix( surfaceTexture, vec4(coloratmoscatter1, 1.0), pow(addedSunsetRim,8.0) );//*addedcloudShadow;

	//add sunset color to clouds
	cloudTexture =  mix( cloudTexture, vec4(coloratmoscatter1, 1.0), pow(addedSunsetRim,6.0) );

	//-----------------
	//process lambert
	
	//--cloud shadows


	vec4 surfaceDiffuse = vec4(addedLambertBump, 1.0)*surfaceTexture*addedcloudShadow;

	vec4 atmoHazeDiffuse = vec4(coloratmo*addedLambert, 1.0);
	surfaceDiffuse = mix(surfaceDiffuse, atmoHazeDiffuse, faceRatioInv);


	vec4 cloudDiffuse = vec4(addedLambert, 1.0)*cloudTexture;

	surfaceDiffuse = mix( surfaceDiffuse, cloudDiffuse,  cloudTexture.a );

	
	//-----------------
	//Atmo scatter
	vec4 atmoNightScatter = vec4(addedCrepuscularScatter*coloratmoscatter2*0.1, 1.0);

	vec4 atmoScatter = vec4(addedSpecAtmo*coloratmoscatter1, 1.0);


	//-----------------
	//Specular

	vec4 surfaceSpec = vec4(addedSpecSurface, 1.0)*texture2D(txtrspec, vUv)*cloudAlphaInv*0.25*addedcloudShadow;

	vec4 cloudSpec = vec4(addedSpecCloud, 1.0)*(cloudTexture.a);

	

	//-----------------
	//Glow

	vec4 lights = mix(texture2D(txtrlights, vUv), texture2D(txtrlightsscatter, vUv), cloudAlpha);

	float lightVal = clamp(pow(addedLightMask,5.0),0.0,1.0);
	lightVal = clamp( texture2D(txtrlightsoffset, vUv).r + (( lightVal * 2.0)-1.0), 0.0, 1.0 );
	//lightVal = (floor(lightVal) + lightVal)*0.5;
	lightVal = floor(lightVal)*0.7 + lightVal*0.3;
	//glowNight *= vec4(1.0 - pow(addedLambert, 8.0), 1.0);
	lights = lights*lightVal;





	
	gl_FragColor = surfaceDiffuse;

	//gl_FragColor = vec4(lightVal);

	////gl_FragColor = texture2D(txtrclouds, vUv);
	
	//gl_FragColor += vec4(addedSunsetRim,addedSunsetRim,addedSunsetRim,1.0); 

	gl_FragColor += lights;

	gl_FragColor += atmoNightScatter;

	gl_FragColor += surfaceSpec;
	gl_FragColor += cloudSpec;
	gl_FragColor += atmoScatter;

}