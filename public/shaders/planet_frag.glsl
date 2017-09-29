
#define M_PI 3.1415926535897932384626433832795

#extension GL_OES_standard_derivatives : enable

const highp float;

uniform sampler2D map_diff;
uniform sampler2D map_spec;

uniform sampler2D map_bump;
uniform float bumpScale;

uniform sampler2D map_clouds;
uniform sampler2D map_cloud_disp;
uniform float cloud_height;
uniform float cloud_disp;
uniform vec3  color_cloud_shadow;

uniform vec3 color_atmo;
uniform vec3 color_atmoscatter_sunset;
uniform vec3 color_atmoscatter_night;

uniform sampler2D map_lights;
uniform sampler2D map_lights_scatter;
uniform sampler2D map_lights_offset;

// uniform sampler2D map_glow;
// uniform sampler2D map_glowscatter;


//uniform vec3   hovercolor;


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


//--------------------------
//From ThreeJS Bump frag
//vvvvvvvvvvvvvvvvvvvvvvvvvv
vec2 dHdxy_fwd() {

	vec2 dSTdx = dFdx( vUv );
	vec2 dSTdy = dFdy( vUv );

	float Hll = bumpScale * texture2D( map_bump, vUv ).x;
	float dBx = bumpScale * texture2D( map_bump, vUv + dSTdx ).x - Hll;
	float dBy = bumpScale * texture2D( map_bump, vUv + dSTdy ).x - Hll;

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
//^^^^^^^^^^^^^^^^^^^^^^^^^
//--------------------------

void main() {

	//----------
	vec3 viewNormalBump = viewNormal;
	viewNormalBump = perturbNormalArb( viewVertPos, viewNormalBump, dHdxy_fwd() );

	vec3 worldViewDirection = normalize(cameraPosition - worldVertPos);
	vec3 cameraViewDirection = normalize(viewVertPos); 

	float faceRatio = clamp( -dot(viewNormal,cameraViewDirection), 0.0, 1.0 );
	float faceRatioInv = 1.0 - faceRatio;

	//-----------------
	//---Clouds and Cloud Displacement. 

	float offset = sin( cloud_disp * 2.0 * M_PI )*0.2;

	vec4 cloudDispTexture = texture2D(map_cloud_disp, vUv);

	vec2 vUv_offset = vUv + vec2(offset*(cloudDispTexture.g-0.5), offset*(cloudDispTexture.r-0.5));
	vUv_offset = mod(vUv_offset,1.0);

	vec4 cloudTexture = texture2D(map_clouds, vUv_offset);
	

	float cloudAlpha = cloudTexture.a;
	float cloudAlphaInv = (1.0 - cloudTexture.a);


	float cloudShadowDisp = cloud_height / 10000.0;
	
	//float eye_distance = length(viewVertPos);

	//-----------------------------
	//ThreeJS fix
	// Workaround for Adreno 3XX dFd*( vec3 ) bug. See #9988
	//detla pos
	vec3 q0 = vec3( dFdx( -viewVertPos.x ), dFdx( -viewVertPos.y ), dFdx( -viewVertPos.z ) );
	vec3 q1 = vec3( dFdy( -viewVertPos.x ), dFdy( -viewVertPos.y ), dFdy( -viewVertPos.z ) );

	//delta uv
	vec2 st0 = dFdx( vUv.st );
	vec2 st1 = dFdy( vUv.st );

	vec3 bitanU = normalize( q0 * st1.t - q1 * st0.t );
	vec3 bitanV = normalize( -q0 * st1.s + q1 * st0.s );

	//-----------------------------
	//Light calculations
	
	vec3 addedSpecAtmo    = vec3(0,0,0);
	vec3 addedSpecCloud   = vec3(0,0,0);
	vec3 addedSpecSurface = vec3(0,0,0);

	vec3 addedLambert = vec3(0,0,0);
	vec3 addedLambertBump = vec3(0,0,0);

	float addedLightMask = 0.0;
	float addedSunsetRim = 0.0;

	vec3 addedCrepuscularScatter = vec3(0,0,0);

	float addedcloudShadow = 0.0;

	//added_normal = length(T);

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
			//calc shadow offset
			float shadowOffsetU = dot( lightDirection , bitanU)*cloudShadowDisp;
			float shadowOffsetV = dot( lightDirection , bitanV)*cloudShadowDisp;

			vec2 vUv_shadow_offset = vec2(vUv_offset.s - shadowOffsetU, vUv_offset.t - shadowOffsetV);
			vUv_shadow_offset = mod(vUv_shadow_offset,1.0);

			addedcloudShadow +=  (1.0 - texture2D( map_clouds, vUv_shadow_offset ).a);

			//---Spec
			float specular = clamp(dot(reflect(lightDirection, viewNormal), cameraViewDirection),0.0, 1.0); 
			float specularBump = clamp(dot(reflect(lightDirection, viewNormalBump), cameraViewDirection),0.0, 1.0);


			float specular_atmo = pow(specular, 10.0)*pow(faceRatioInv, 4.0);
			float specular_cloud = pow(specular, 4.0)*pow(faceRatio, 1.5);
			float specular_surface = pow(specularBump , 10.0)*pow(faceRatio, 1.5);

			addedSpecAtmo += specular_atmo * directionalLights[ i ].color;
			addedSpecCloud += specular_cloud * directionalLights[ i ].color;
			addedSpecSurface += specular_surface * directionalLights[ i ].color;
		}
	#endif



	//-----------------

	//add atmosphere haze based on facing ratio
	
	vec4 surfaceTexture = texture2D(map_diff, vUv);

	//add sunset color to surface
	surfaceTexture =  mix( surfaceTexture, vec4(color_atmoscatter_sunset, 1.0), pow(addedSunsetRim,10.0) );

	//add sunset color to clouds

	cloudTexture =  mix( cloudTexture, vec4(color_atmoscatter_sunset, cloudTexture.a), pow(addedSunsetRim,8.0) );

	//-----------------
	//process lambert
	
	//--cloud shadows
	vec4 cloudShadowColor = mix( vec4(color_cloud_shadow,1.0), vec4(color_atmoscatter_sunset,1.0), pow(addedSunsetRim,8.0));

	vec4 cloudShadow = 1.0 - (1.0 - cloudShadowColor)*(1.0 - addedcloudShadow);
	vec4 surfaceDiffuse = vec4(addedLambertBump, 1.0)*surfaceTexture*cloudShadow;

	vec4 atmoHazeDiffuse = vec4(color_atmo*addedLambert, 1.0);
	surfaceDiffuse = mix(surfaceDiffuse, atmoHazeDiffuse, faceRatioInv);

	vec4 cloudDiffuse = vec4(addedLambert, 1.0)*cloudTexture;

	surfaceDiffuse = mix( surfaceDiffuse, cloudDiffuse,  cloudTexture.a );

	//-----------------
	//Atmo scatter
	vec4 atmoNightScatter = vec4(addedCrepuscularScatter*color_atmoscatter_night*0.05, 1.0);

	vec4 atmoScatter = vec4(addedSpecAtmo*color_atmoscatter_sunset, 1.0)*0.5;

	//-----------------
	//Specular

	vec4 surfaceSpec = vec4(addedSpecSurface, 1.0)*texture2D(map_spec, vUv)*cloudAlphaInv*addedcloudShadow*0.15;

	vec4 cloudSpec = vec4(addedSpecCloud, 1.0)*(cloudTexture.a)*0.3;

	//-----------------
	//Glow

	vec4 lights = mix(texture2D(map_lights, vUv), texture2D(map_lights_scatter, vUv), cloudAlpha);

	float lightVal = clamp(pow(addedLightMask,5.0),0.0,1.0);
	lightVal = clamp( texture2D(map_lights_offset, vUv).r + (( lightVal * 2.0)-1.0), 0.0, 1.0 );
	lightVal = floor(lightVal)*0.7 + lightVal*0.3;
	lights = lights*lightVal;

	//gl_FragColor = cloudShadowColor;	

	gl_FragColor = surfaceDiffuse;
	gl_FragColor += lights;
	gl_FragColor += atmoNightScatter;
	gl_FragColor += surfaceSpec;
	gl_FragColor += cloudSpec;
	gl_FragColor += atmoScatter;

}