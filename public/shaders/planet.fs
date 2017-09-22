<script type="x-shader/x-fragment" id="shader">



	precision mediump float; 

	varying vec2 vUv;
	varying vec3 worldNormal;
	varying vec3 worldVertPos;

	varying vec3 viewNormal;
	varying vec3 viewVertPos;

	varying vec3 vViewPosition;//?

	
	uniform sampler2D txtrdiff;
	uniform sampler2D txtrspec;
	uniform sampler2D txtrclouds;
	uniform sampler2D txtrclouddisp;
	uniform sampler2D txtrglowday;
	uniform sampler2D txtrglownight;
	uniform sampler2D txtrglowscatter;

	uniform vec3   atmoColor;
	uniform vec3   hoverColor;
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



	uniform vec3 ambientLightColor;
	

	vec3 multiplyColor(vec3 _color1, vec3 _color2)
	{
		_color1.r*=_color2.r;
		_color1.g*=_color2.g;
		_color1.b*=_color2.b;

		return _color1;
	}

	vec4 multiplyColor(vec4 _color1, vec4 _color2)
	{
		_color1.r*=_color2.r;
		_color1.g*=_color2.g;
		_color1.b*=_color2.b;

		return _color1;
	}



	void main() {

		
		
		//---Clouds and Cloud Displacement. 
		vec2 vUv_offset1 = vUv;
		vec2 vUv_offset2 = vUv;

		vec4 cloudDispTexture = texture2D(txtrclouddisp, vUv);
		float clouddispscale2= mod(clouddispscale+0.5,1.0);

		vec2 cloudDisp1 = vec2(clouddispscale*(cloudDispTexture.g-0.5)*0.4, clouddispscale*(cloudDispTexture.r-0.5)*0.4);
		vec2 cloudDisp2 = vec2(clouddispscale2*(cloudDispTexture.g-0.5)*0.4, clouddispscale2*(cloudDispTexture.r-0.5)*0.4);

		vUv_offset1 = mod(vUv_offset1+cloudDisp1,1.0);
		vUv_offset2 = mod(vUv_offset2+cloudDisp2,1.0);

		vec4 cloudTexture1 = texture2D(txtrclouds, vUv_offset1);
		vec4 cloudTexture2 = texture2D(txtrclouds, vUv_offset2);

		float cloudMix = clamp(abs(-8.0*clouddispscale+4.0)-1.0, 0.0,1.0);


		vec4 cloudTexture = mix(cloudTexture1, cloudTexture2,cloudMix);

		 
		//---Lighting
		
		//vec3 viewDir    = normalize(cameraPosition - worldVertPos);
		
		vec3 viewDir    = normalize(cameraPosition - viewVertPos);
	
		float faceRatio = max(dot(viewDir,viewNormal), 0.0);
		
		vec3 specTexture  = texture2D(txtrspec, vUv).rgb;

		vec4 addedLambert  = vec4(0.0,0.0,0.0, 1.0);
		vec4 addedSpecular = vec4(0.0,0.0,0.0, 1.0);

		vec4 addedCloudLambert = vec4(0.0,0.0,0.0, 1.0);
		vec4 addedCloudSpecular = vec4(0.0,0.0,0.0, 1.0);	

		vec4 addedAtmoGlow = vec4(0.0,0.0,0.0, 1.0);
		
		

		//--- Sum Directional Lights. 
		#if NUM_DIR_LIGHTS > 0
		for(int i = 0; i < NUM_DIR_LIGHTS; i++) {
			
			
			vec3  lightDir   = normalize( directionalLights[ i ].direction); 

			//---Diff
			float lambert     = dot(lightDir,worldNormal);
			addedLambert.rgb += clamp(lambert*directionalLights[i].color, 0.0, 1.0);

			//---Surface-Spec
			float specScale = clamp(dot(reflect(viewNormal, worldNormal), lightDir) ,0.0, 1.0);
			
			vec3  lightSpec = directionalLights[i].color*pow(specScale, 10.0)*faceRatio*(1.0-cloudTexture.a);
			lightSpec = multiplyColor(lightSpec,specTexture);

			//---Cloud-Spec
			vec3 cloudSpec = 0.3*directionalLights[i].color*pow(specScale, 2.0)*cloudTexture.a*faceRatio;
			cloudSpec = multiplyColor(cloudSpec,cloudTexture.rgb);
	
			addedSpecular.rgb += lightSpec+cloudSpec;
			
			
		}
		#endif


		//--- Sum Point Lights. 
		#if NUM_POINT_LIGHTS > 0
		for(int i = 0; i < NUM_POINT_LIGHTS; i++) {
			/*
			vec3  relativeLightPosition = worldVertPos-pointLights[i].position;
			vec3  lightDir              = normalize(relativeLightPosition);
			float relativeLightDistance = length(relativeLightPosition);


			
			//--- Quadradic falloff
			//float lightBrightness = pointLightDistance[i]/(relativeLightDistance*relativeLightDistance);
			
			//-- Linear Falloff to match THREE.js definition of point lights. 
			float lightBrightness = clamp(1.0-(relativeLightDistance/pointLightDistance[i]), 0.0, 1.0); 

			//---Diff
			float lambert = dot(-lightDir,worldNormal);
			addedLambert.rgb += clamp(lambert*pointLightColor[i]*lightBrightness, 0.0, 1.0);

			//---Surface-Spec
			float specScale = clamp(dot(reflect(viewDir, worldNormal), lightDir) ,0.0, 1.0);
			vec3  lightSpec = pointLightColor[i]*pow(specScale, 10.0)*faceRatio*(1.0-cloudTexture.a)*lightBrightness;
			lightSpec = multiplyColor(lightSpec,specTexture);

			//---Cloud-Spec
			vec3 cloudSpec = 0.3*pointLightColor[i]*pow(specScale, 2.0)*cloudTexture.a*faceRatio*lightBrightness;
			cloudSpec = multiplyColor(cloudSpec,cloudTexture.rgb);
	
			addedSpecular.rgb += lightSpec+cloudSpec;
			*/
		}
		#endif


		vec4 atmoGlow = 2.0*vec4(atmoColor,0.0)*(1.0-faceRatio);
		atmoGlow.r *= atmoColor.r*addedLambert.r;
		atmoGlow.g *= atmoColor.g*addedLambert.g;
		atmoGlow.b *= atmoColor.b*addedLambert.b;
		addedAtmoGlow += clamp(atmoGlow,0.0,1.0);


		//---glow maps
		vec4 glowDay = texture2D(txtrglowday, vUv);
		glowDay = multiplyColor(glowDay,addedLambert)*faceRatio;
		glowDay *= 1.0-cloudTexture.a;
		glowDay *= 1.0-cloudTexture.a;
		glowDay *= 1.0-cloudTexture.a;

		vec4 glowNight = texture2D(txtrglownight, vUv);
		glowNight*=faceRatio;

		glowNight.r *= 1.0-cloudTexture.a;
		glowNight.g *= 1.0-cloudTexture.a;
		glowNight.b *= 1.0-cloudTexture.a;

		vec4 glowScatterRaw = texture2D(txtrglowscatter, vUv);

		vec4 glowScatter = glowScatterRaw;

		//add scatter into clouds
		glowScatter.r *= cloudTexture.a;
		glowScatter.g *= cloudTexture.a;
		glowScatter.b *= cloudTexture.a;

		//add scatter into edge of atmosphere
		glowScatter += glowScatterRaw * (1.0-faceRatio);

		//add scatter to night glow 
		glowNight += glowScatter;

	
		//multiply inverse of day to get only night
		glowNight.r *= pow(1.0-addedLambert.r,2.0);  
		glowNight.g *= pow(1.0-addedLambert.g,2.0);  
		glowNight.b *= pow(1.0-addedLambert.b,2.0); 
		

		addedLambert.rgb += ambientLightColor;


		vec4 totalDiff = mix( texture2D(txtrdiff, vUv), cloudTexture, cloudTexture.a);
		totalDiff = mix( totalDiff, vec4(atmoColor,0.0), 1.0-faceRatio);

		//gl_FragColor = (totalDiff * addedLambert)+addedSpecular+addedAtmoGlow+glowDay+glowNight;
		//gl_FragColor += vec4(hoverColor, 0.0);
		gl_FragColor = addedSpecular;
		gl_FragColor = clamp(gl_FragColor,0.0,1.0);

	}



</script>