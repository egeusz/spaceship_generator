// 1: precision highp float;
// 2: precision highp int;
// 3: #define SHADER_NAME ShaderMaterial
// 4: #define GAMMA_FACTOR 2
// 5: #define NUM_CLIPPING_PLANES 0
// 6: #define UNION_CLIPPING_PLANES 0
// 7: uniform mat4 viewMatrix;
// 8: uniform vec3 cameraPosition;
// 9: #define TONE_MAPPING
// 10: #define saturate(a) clamp( a, 0.0, 1.0 )
// 11: uniform float toneMappingExposure;
// 12: uniform float toneMappingWhitePoint;
// 13: vec3 LinearToneMapping( vec3 color ) {
// 14: 	return toneMappingExposure * color;
// 15: }
// 16: vec3 ReinhardToneMapping( vec3 color ) {
// 17: 	color *= toneMappingExposure;
// 18: 	return saturate( color / ( vec3( 1.0 ) + color ) );
// 19: }
// 20: #define Uncharted2Helper( x ) max( ( ( x * ( 0.15 * x + 0.10 * 0.50 ) + 0.20 * 0.02 ) / ( x * ( 0.15 * x + 0.50 ) + 0.20 * 0.30 ) ) - 0.02 / 0.30, vec3( 0.0 ) )
// 21: vec3 Uncharted2ToneMapping( vec3 color ) {
// 22: 	color *= toneMappingExposure;
// 23: 	return saturate( Uncharted2Helper( color ) / Uncharted2Helper( vec3( toneMappingWhitePoint ) ) );
// 24: }
// 25: vec3 OptimizedCineonToneMapping( vec3 color ) {
// 26: 	color *= toneMappingExposure;
// 27: 	color = max( vec3( 0.0 ), color - 0.004 );
// 28: 	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
// 29: }
// 30: 
// 31: vec3 toneMapping( vec3 color ) { return LinearToneMapping( color ); }
// 32: 
// 33: vec4 LinearToLinear( in vec4 value ) {
// 34: 	return value;
// 35: }
// 36: vec4 GammaToLinear( in vec4 value, in float gammaFactor ) {
// 37: 	return vec4( pow( value.xyz, vec3( gammaFactor ) ), value.w );
// 38: }
// 39: vec4 LinearToGamma( in vec4 value, in float gammaFactor ) {
// 40: 	return vec4( pow( value.xyz, vec3( 1.0 / gammaFactor ) ), value.w );
// 41: }
// 42: vec4 sRGBToLinear( in vec4 value ) {
// 43: 	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.w );
// 44: }
// 45: vec4 LinearTosRGB( in vec4 value ) {
// 46: 	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.w );
// 47: }
// 48: vec4 RGBEToLinear( in vec4 value ) {
// 49: 	return vec4( value.rgb * exp2( value.a * 255.0 - 128.0 ), 1.0 );
// 50: }
// 51: vec4 LinearToRGBE( in vec4 value ) {
// 52: 	float maxComponent = max( max( value.r, value.g ), value.b );
// 53: 	float fExp = clamp( ceil( log2( maxComponent ) ), -128.0, 127.0 );
// 54: 	return vec4( value.rgb / exp2( fExp ), ( fExp + 128.0 ) / 255.0 );
// 55: }
// 56: vec4 RGBMToLinear( in vec4 value, in float maxRange ) {
// 57: 	return vec4( value.xyz * value.w * maxRange, 1.0 );
// 58: }
// 59: vec4 LinearToRGBM( in vec4 value, in float maxRange ) {
// 60: 	float maxRGB = max( value.x, max( value.g, value.b ) );
// 61: 	float M      = clamp( maxRGB / maxRange, 0.0, 1.0 );
// 62: 	M            = ceil( M * 255.0 ) / 255.0;
// 63: 	return vec4( value.rgb / ( M * maxRange ), M );
// 64: }
// 65: vec4 RGBDToLinear( in vec4 value, in float maxRange ) {
// 66: 	return vec4( value.rgb * ( ( maxRange / 255.0 ) / value.a ), 1.0 );
// 67: }
// 68: vec4 LinearToRGBD( in vec4 value, in float maxRange ) {
// 69: 	float maxRGB = max( value.x, max( value.g, value.b ) );
// 70: 	float D      = max( maxRange / maxRGB, 1.0 );
// 71: 	D            = min( floor( D ) / 255.0, 1.0 );
// 72: 	return vec4( value.rgb * ( D * ( 255.0 / maxRange ) ), D );
// 73: }
// 74: const mat3 cLogLuvM = mat3( 0.2209, 0.3390, 0.4184, 0.1138, 0.6780, 0.7319, 0.0102, 0.1130, 0.2969 );
// 75: vec4 LinearToLogLuv( in vec4 value )  {
// 76: 	vec3 Xp_Y_XYZp = value.rgb * cLogLuvM;
// 77: 	Xp_Y_XYZp = max(Xp_Y_XYZp, vec3(1e-6, 1e-6, 1e-6));
// 78: 	vec4 vResult;
// 79: 	vResult.xy = Xp_Y_XYZp.xy / Xp_Y_XYZp.z;
// 80: 	float Le = 2.0 * log2(Xp_Y_XYZp.y) + 127.0;
// 81: 	vResult.w = fract(Le);
// 82: 	vResult.z = (Le - (floor(vResult.w*255.0))/255.0)/255.0;
// 83: 	return vResult;
// 84: }
// 85: const mat3 cLogLuvInverseM = mat3( 6.0014, -2.7008, -1.7996, -1.3320, 3.1029, -5.7721, 0.3008, -1.0882, 5.6268 );
// 86: vec4 LogLuvToLinear( in vec4 value ) {
// 87: 	float Le = value.z * 255.0 + value.w;
// 88: 	vec3 Xp_Y_XYZp;
// 89: 	Xp_Y_XYZp.y = exp2((Le - 127.0) / 2.0);
// 90: 	Xp_Y_XYZp.z = Xp_Y_XYZp.y / value.y;
// 91: 	Xp_Y_XYZp.x = value.x * Xp_Y_XYZp.z;
// 92: 	vec3 vRGB = Xp_Y_XYZp.rgb * cLogLuvInverseM;
// 93: 	return vec4( max(vRGB, 0.0), 1.0 );
// 94: }
// 95: 
// 96: vec4 mapTexelToLinear( vec4 value ) { return LinearToLinear( value ); }
// 97: vec4 envMapTexelToLinear( vec4 value ) { return LinearToLinear( value ); }
// 98: vec4 emissiveMapTexelToLinear( vec4 value ) { return LinearToLinear( value ); }
// 99: vec4 linearToOutputTexel( vec4 value ) { return LinearToLinear( value ); }
// 100: 
#define PHONG

uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;

//#include <common>
	#define PI 3.14159265359
	#define PI2 6.28318530718
	#define PI_HALF 1.5707963267949
	#define RECIPROCAL_PI 0.31830988618
	#define RECIPROCAL_PI2 0.15915494
	#define LOG2 1.442695
	#define EPSILON 1e-6

	#define saturate(a) clamp( a, 0.0, 1.0 )
	#define whiteCompliment(a) ( 1.0 - saturate( a ) )

	float pow2( const in float x ) { return x*x; }
	float pow3( const in float x ) { return x*x*x; }
	float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
	float average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }
	// expects values in the range of [0,1]x[0,1], returns values in the [0,1] range.
	// do not collapse into a single function per: http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
	highp float rand( const in vec2 uv ) {
		const highp float a = 12.9898, b = 78.233, c = 43758.5453;
		highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
		return fract(sin(sn) * c);
	}

	struct IncidentLight {
		vec3 color;
		vec3 direction;
		bool visible;
	};

	struct ReflectedLight {
		vec3 directDiffuse;
		vec3 directSpecular;
		vec3 indirectDiffuse;
		vec3 indirectSpecular;
	};

	struct GeometricContext {
		vec3 position;
		vec3 normal;
		vec3 viewDir;
	};

	vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

		return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

	}

	// http://en.wikibooks.org/wiki/GLSL_Programming/Applying_Matrix_Transformations
	vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {

		return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );

	}

	vec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {

		float distance = dot( planeNormal, point - pointOnPlane );

		return - distance * planeNormal + point;

	}

	float sideOfPlane( in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {

		return sign( dot( point - pointOnPlane, planeNormal ) );

	}

	vec3 linePlaneIntersect( in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal ) {

		return lineDirection * ( dot( planeNormal, pointOnPlane - pointOnLine ) / dot( planeNormal, lineDirection ) ) + pointOnLine;

	}

	mat3 transpose( const in mat3 v ) {

		mat3 tmp;
		tmp[0] = vec3(v[0].x, v[1].x, v[2].x);
		tmp[1] = vec3(v[0].y, v[1].y, v[2].y);
		tmp[2] = vec3(v[0].z, v[1].z, v[2].z);

		return tmp;

	}

//#include <packing>
//packing
	vec3 packNormalToRGB( const in vec3 normal ) {
		return normalize( normal ) * 0.5 + 0.5;
	}

	vec3 unpackRGBToNormal( const in vec3 rgb ) {
		return 1.0 - 2.0 * rgb.xyz;
	}

	const float PackUpscale = 256. / 255.; // fraction -> 0..1 (including 1)
	const float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)

	const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256.,  256. );
	const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );

	const float ShiftRight8 = 1. / 256.;

	vec4 packDepthToRGBA( const in float v ) {
		vec4 r = vec4( fract( v * PackFactors ), v );
		r.yzw -= r.xyz * ShiftRight8; // tidy overflow
		return r * PackUpscale;
	}

	float unpackRGBAToDepth( const in vec4 v ) {
		return dot( v, UnpackFactors );
	}

	// NOTE: viewZ/eyeZ is < 0 when in front of the camera per OpenGL conventions

	float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
		return ( viewZ + near ) / ( near - far );
	}
	float orthographicDepthToViewZ( const in float linearClipZ, const in float near, const in float far ) {
		return linearClipZ * ( near - far ) - near;
	}

	float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
		return (( near + viewZ ) * far ) / (( far - near ) * viewZ );
	}
	float perspectiveDepthToViewZ( const in float invClipZ, const in float near, const in float far ) {
		return ( near * far ) / ( ( far - near ) * invClipZ - far );
	}

//#include <uv_pars_fragment>
#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )

	varying vec2 vUv;
#endif

//#include <uv2_pars_fragment>
#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )

	varying vec2 vUv2;
#endif

//#include <map_pars_fragment>
#ifdef USE_MAP

	uniform sampler2D map;
#endif



//#include <bsdfs>
//BSDFS
	float punctualLightIntensityToIrradianceFactor( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {

		if( decayExponent > 0.0 ) {

	#if defined ( PHYSICALLY_CORRECT_LIGHTS )

			// based upon Frostbite 3 Moving to Physically-based Rendering
			// page 32, equation 26: E[window1]
			// http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr_v2.pdf
			// this is intended to be used on spot and point lights who are represented as luminous intensity
			// but who must be converted to luminous irradiance for surface lighting calculation
			float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
			float maxDistanceCutoffFactor = pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
			return distanceFalloff * maxDistanceCutoffFactor;

	#else

			return pow( saturate( -lightDistance / cutoffDistance + 1.0 ), decayExponent );

	#endif

		}

		return 1.0;

	}

	vec3 BRDF_Diffuse_Lambert( const in vec3 diffuseColor ) {

		return RECIPROCAL_PI * diffuseColor;

	} // validated

	vec3 F_Schlick( const in vec3 specularColor, const in float dotLH ) {

		// Original approximation by Christophe Schlick '94
		// float fresnel = pow( 1.0 - dotLH, 5.0 );

		// Optimized variant (presented by Epic at SIGGRAPH '13)
		float fresnel = exp2( ( -5.55473 * dotLH - 6.98316 ) * dotLH );

		return ( 1.0 - specularColor ) * fresnel + specularColor;

	} // validated

	// Microfacet Models for Refraction through Rough Surfaces - equation (34)
	// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
	// alpha is "roughness squared" in Disney’s reparameterization
	float G_GGX_Smith( const in float alpha, const in float dotNL, const in float dotNV ) {

		// geometry term = G(l)⋅G(v) / 4(n⋅l)(n⋅v)

		float a2 = pow2( alpha );

		float gl = dotNL + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
		float gv = dotNV + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );

		return 1.0 / ( gl * gv );

	} // validated

	// Moving Frostbite to Physically Based Rendering 2.0 - page 12, listing 2
	// http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr_v2.pdf
	float G_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {

		float a2 = pow2( alpha );

		// dotNL and dotNV are explicitly swapped. This is not a mistake.
		float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
		float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );

		return 0.5 / max( gv + gl, EPSILON );
	}

	// Microfacet Models for Refraction through Rough Surfaces - equation (33)
	// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
	// alpha is "roughness squared" in Disney’s reparameterization
	float D_GGX( const in float alpha, const in float dotNH ) {

		float a2 = pow2( alpha );

		float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0; // avoid alpha = 0 with dotNH = 1

		return RECIPROCAL_PI * a2 / pow2( denom );

	}

	// GGX Distribution, Schlick Fresnel, GGX-Smith Visibility
	vec3 BRDF_Specular_GGX( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {

		float alpha = pow2( roughness ); // UE4's roughness

		vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );

		float dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );
		float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
		float dotNH = saturate( dot( geometry.normal, halfDir ) );
		float dotLH = saturate( dot( incidentLight.direction, halfDir ) );

		vec3 F = F_Schlick( specularColor, dotLH );

		float G = G_GGX_SmithCorrelated( alpha, dotNL, dotNV );

		float D = D_GGX( alpha, dotNH );

		return F * ( G * D );

	} // validated

	// Rect Area Light

	// Area light computation code adapted from:
	// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
	// By: Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
	// https://drive.google.com/file/d/0BzvWIdpUpRx_d09ndGVjNVJzZjA/view
	// https://eheitzresearch.wordpress.com/415-2/
	// http://blog.selfshadow.com/sandbox/ltc.html

	vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {

		const float LUT_SIZE  = 64.0;
		const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
		const float LUT_BIAS  = 0.5 / LUT_SIZE;

		float theta = acos( dot( N, V ) );

		// Parameterization of texture:
		// sqrt(roughness) -> [0,1]
		// theta -> [0, PI/2]
		vec2 uv = vec2(
			sqrt( saturate( roughness ) ),
			saturate( theta / ( 0.5 * PI ) ) );

		// Ensure we don't have nonlinearities at the look-up table's edges
		// see: http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter24.html
		//      "Shader Analysis" section
		uv = uv * LUT_SCALE + LUT_BIAS;

		return uv;

	}

	// Real-Time Area Lighting: a Journey from Research to Production
	// By: Stephen Hill & Eric Heitz
	// http://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf
	// An approximation for the form factor of a clipped rectangle.
	float LTC_ClippedSphereFormFactor( const in vec3 f ) {

		float l = length( f );

		return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );

	}

	// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
	// also Real-Time Area Lighting: a Journey from Research to Production
	// http://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf
	// Normalization by 2*PI is incorporated in this function itself.
	// theta/sin(theta) is approximated by rational polynomial
	vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {

		float x = dot( v1, v2 );

		float y = abs( x );
		float a = 0.86267 + (0.49788 + 0.01436 * y ) * y;
		float b = 3.45068 + (4.18814 + y) * y;
		float v = a / b;

		float theta_sintheta = (x > 0.0) ? v : 0.5 * inversesqrt( 1.0 - x * x ) - v;

		return cross( v1, v2 ) * theta_sintheta;

	}

	vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {

		// bail if point is on back side of plane of light
		// assumes ccw winding order of light vertices
		vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
		vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
		vec3 lightNormal = cross( v1, v2 );

		if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );

		// construct orthonormal basis around N
		vec3 T1, T2;
		T1 = normalize( V - N * dot( V, N ) );
		T2 = - cross( N, T1 ); // negated from paper; possibly due to a different assumed handedness of world coordinate system

		// compute transform
		mat3 mat = mInv * transpose( mat3( T1, T2, N ) );

		// transform rect
		vec3 coords[ 4 ];
		coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
		coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
		coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
		coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );

		// project rect onto sphere
		coords[ 0 ] = normalize( coords[ 0 ] );
		coords[ 1 ] = normalize( coords[ 1 ] );
		coords[ 2 ] = normalize( coords[ 2 ] );
		coords[ 3 ] = normalize( coords[ 3 ] );

		// calculate vector form factor
		vec3 vectorFormFactor = vec3( 0.0 );
		vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
		vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
		vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
		vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );

		// adjust for horizon clipping
		vec3 result = vec3( LTC_ClippedSphereFormFactor( vectorFormFactor ) );

		return result;

	}

	// End Rect Area Light

	// ref: https://www.unrealengine.com/blog/physically-based-shading-on-mobile - environmentBRDF for GGX on mobile
	vec3 BRDF_Specular_GGX_Environment( const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {

		float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );

		const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );

		const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );

		vec4 r = roughness * c0 + c1;

		float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;

		vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;

		return specularColor * AB.x + AB.y;

	} // validated


	float G_BlinnPhong_Implicit( /* const in float dotNL, const in float dotNV */ ) {

		// geometry term is (n dot l)(n dot v) / 4(n dot l)(n dot v)
		return 0.25;

	}

	float D_BlinnPhong( const in float shininess, const in float dotNH ) {

		return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );

	}

	vec3 BRDF_Specular_BlinnPhong( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float shininess ) {

		vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );

		//float dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );
		//float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
		float dotNH = saturate( dot( geometry.normal, halfDir ) );
		float dotLH = saturate( dot( incidentLight.direction, halfDir ) );

		vec3 F = F_Schlick( specularColor, dotLH );

		float G = G_BlinnPhong_Implicit( /* dotNL, dotNV */ );

		float D = D_BlinnPhong( shininess, dotNH );

		return F * ( G * D );

	} // validated

	// source: http://simonstechblog.blogspot.ca/2011/12/microfacet-brdf.html
	float GGXRoughnessToBlinnExponent( const in float ggxRoughness ) {
		return ( 2.0 / pow2( ggxRoughness + 0.0001 ) - 2.0 );
	}

	float BlinnExponentToGGXRoughness( const in float blinnExponent ) {
		return sqrt( 2.0 / ( blinnExponent + 2.0 ) );
	}

//#include <lights_pars>
//LIGHTS
	uniform vec3 ambientLightColor;

	vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {

		vec3 irradiance = ambientLightColor;

		#ifndef PHYSICALLY_CORRECT_LIGHTS

			irradiance *= PI;

		#endif

		return irradiance;

	}

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

		void getDirectionalDirectLightIrradiance( const in DirectionalLight directionalLight, const in GeometricContext geometry, out IncidentLight directLight ) {

			directLight.color = directionalLight.color;
			directLight.direction = directionalLight.direction;
			directLight.visible = true;

		}

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

		// directLight is an out parameter as having it as a return value caused compiler errors on some devices
		void getPointDirectLightIrradiance( const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight directLight ) {

			vec3 lVector = pointLight.position - geometry.position;
			directLight.direction = normalize( lVector );

			float lightDistance = length( lVector );

			directLight.color = pointLight.color;
			directLight.color *= punctualLightIntensityToIrradianceFactor( lightDistance, pointLight.distance, pointLight.decay );
			directLight.visible = ( directLight.color != vec3( 0.0 ) );

		}

	#endif


	#if NUM_SPOT_LIGHTS > 0

		struct SpotLight {
			vec3 position;
			vec3 direction;
			vec3 color;
			float distance;
			float decay;
			float coneCos;
			float penumbraCos;

			int shadow;
			float shadowBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};

		uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];

		// directLight is an out parameter as having it as a return value caused compiler errors on some devices
		void getSpotDirectLightIrradiance( const in SpotLight spotLight, const in GeometricContext geometry, out IncidentLight directLight  ) {

			vec3 lVector = spotLight.position - geometry.position;
			directLight.direction = normalize( lVector );

			float lightDistance = length( lVector );
			float angleCos = dot( directLight.direction, spotLight.direction );

			if ( angleCos > spotLight.coneCos ) {

				float spotEffect = smoothstep( spotLight.coneCos, spotLight.penumbraCos, angleCos );

				directLight.color = spotLight.color;
				directLight.color *= spotEffect * punctualLightIntensityToIrradianceFactor( lightDistance, spotLight.distance, spotLight.decay );
				directLight.visible = true;

			} else {

				directLight.color = vec3( 0.0 );
				directLight.visible = false;

			}
		}

	#endif


	#if NUM_RECT_AREA_LIGHTS > 0

		struct RectAreaLight {
			vec3 color;
			vec3 position;
			vec3 halfWidth;
			vec3 halfHeight;
		};

		// Pre-computed values of LinearTransformedCosine approximation of BRDF
		// BRDF approximation Texture is 64x64
		uniform sampler2D ltcMat; // RGBA Float
		uniform sampler2D ltcMag; // Alpha Float (only has w component)

		uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];

	#endif


	#if NUM_HEMI_LIGHTS > 0

		struct HemisphereLight {
			vec3 direction;
			vec3 skyColor;
			vec3 groundColor;
		};

		uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];

		vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in GeometricContext geometry ) {

			float dotNL = dot( geometry.normal, hemiLight.direction );
			float hemiDiffuseWeight = 0.5 * dotNL + 0.5;

			vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );

			#ifndef PHYSICALLY_CORRECT_LIGHTS

				irradiance *= PI;

			#endif

			return irradiance;

		}

	#endif


	#if defined( USE_ENVMAP ) && defined( PHYSICAL )

		vec3 getLightProbeIndirectIrradiance( /*const in SpecularLightProbe specularLightProbe,*/ const in GeometricContext geometry, const in int maxMIPLevel ) {

			vec3 worldNormal = inverseTransformDirection( geometry.normal, viewMatrix );

			#ifdef ENVMAP_TYPE_CUBE

				vec3 queryVec = vec3( flipEnvMap * worldNormal.x, worldNormal.yz );

				// TODO: replace with properly filtered cubemaps and access the irradiance LOD level, be it the last LOD level
				// of a specular cubemap, or just the default level of a specially created irradiance cubemap.

				#ifdef TEXTURE_LOD_EXT

					vec4 envMapColor = textureCubeLodEXT( envMap, queryVec, float( maxMIPLevel ) );

				#else

					// force the bias high to get the last LOD level as it is the most blurred.
					vec4 envMapColor = textureCube( envMap, queryVec, float( maxMIPLevel ) );

				#endif

				envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;

			#elif defined( ENVMAP_TYPE_CUBE_UV )

				vec3 queryVec = vec3( flipEnvMap * worldNormal.x, worldNormal.yz );
				vec4 envMapColor = textureCubeUV( queryVec, 1.0 );

			#else

				vec4 envMapColor = vec4( 0.0 );

			#endif

			return PI * envMapColor.rgb * envMapIntensity;

		}

		// taken from here: http://casual-effects.blogspot.ca/2011/08/plausible-environment-lighting-in-two.html
		float getSpecularMIPLevel( const in float blinnShininessExponent, const in int maxMIPLevel ) {

			//float envMapWidth = pow( 2.0, maxMIPLevelScalar );
			//float desiredMIPLevel = log2( envMapWidth * sqrt( 3.0 ) ) - 0.5 * log2( pow2( blinnShininessExponent ) + 1.0 );

			float maxMIPLevelScalar = float( maxMIPLevel );
			float desiredMIPLevel = maxMIPLevelScalar - 0.79248 - 0.5 * log2( pow2( blinnShininessExponent ) + 1.0 );

			// clamp to allowable LOD ranges.
			return clamp( desiredMIPLevel, 0.0, maxMIPLevelScalar );

		}

		vec3 getLightProbeIndirectRadiance( /*const in SpecularLightProbe specularLightProbe,*/ const in GeometricContext geometry, const in float blinnShininessExponent, const in int maxMIPLevel ) {

			#ifdef ENVMAP_MODE_REFLECTION

				vec3 reflectVec = reflect( -geometry.viewDir, geometry.normal );

			#else

				vec3 reflectVec = refract( -geometry.viewDir, geometry.normal, refractionRatio );

			#endif

			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );

			float specularMIPLevel = getSpecularMIPLevel( blinnShininessExponent, maxMIPLevel );

			#ifdef ENVMAP_TYPE_CUBE

				vec3 queryReflectVec = vec3( flipEnvMap * reflectVec.x, reflectVec.yz );

				#ifdef TEXTURE_LOD_EXT

					vec4 envMapColor = textureCubeLodEXT( envMap, queryReflectVec, specularMIPLevel );

				#else

					vec4 envMapColor = textureCube( envMap, queryReflectVec, specularMIPLevel );

				#endif

				envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;

			#elif defined( ENVMAP_TYPE_CUBE_UV )

				vec3 queryReflectVec = vec3( flipEnvMap * reflectVec.x, reflectVec.yz );
				vec4 envMapColor = textureCubeUV(queryReflectVec, BlinnExponentToGGXRoughness(blinnShininessExponent));

			#elif defined( ENVMAP_TYPE_EQUIREC )

				vec2 sampleUV;
				sampleUV.y = asin( clamp( reflectVec.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
				sampleUV.x = atan( reflectVec.z, reflectVec.x ) * RECIPROCAL_PI2 + 0.5;

				#ifdef TEXTURE_LOD_EXT

					vec4 envMapColor = texture2DLodEXT( envMap, sampleUV, specularMIPLevel );

				#else

					vec4 envMapColor = texture2D( envMap, sampleUV, specularMIPLevel );

				#endif

				envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;

			#elif defined( ENVMAP_TYPE_SPHERE )

				vec3 reflectView = normalize( ( viewMatrix * vec4( reflectVec, 0.0 ) ).xyz + vec3( 0.0,0.0,1.0 ) );

				#ifdef TEXTURE_LOD_EXT

					vec4 envMapColor = texture2DLodEXT( envMap, reflectView.xy * 0.5 + 0.5, specularMIPLevel );

				#else

					vec4 envMapColor = texture2D( envMap, reflectView.xy * 0.5 + 0.5, specularMIPLevel );

				#endif

				envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;

			#endif

			return envMapColor.rgb * envMapIntensity;

		}

	#endif

//#include <lights_phong_pars_fragment>
	varying vec3 vViewPosition;

	#ifndef FLAT_SHADED

		varying vec3 vNormal;

	#endif


	struct BlinnPhongMaterial {

		vec3	diffuseColor;
		vec3	specularColor;
		float	specularShininess;
		float	specularStrength;

	};

	void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {

		#ifdef TOON

			vec3 irradiance = getGradientIrradiance( geometry.normal, directLight.direction ) * directLight.color;

		#else

			float dotNL = saturate( dot( geometry.normal, directLight.direction ) );
			vec3 irradiance = dotNL * directLight.color;

		#endif

		#ifndef PHYSICALLY_CORRECT_LIGHTS

			irradiance *= PI; // punctual light

		#endif

		reflectedLight.directDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );

		reflectedLight.directSpecular += irradiance * BRDF_Specular_BlinnPhong( directLight, geometry, material.specularColor, material.specularShininess ) * material.specularStrength;

	}

	void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {

		reflectedLight.indirectDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );

	}

	#define RE_Direct				RE_Direct_BlinnPhong
	#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong

	#define Material_LightProbeLOD( material )	(0)

//#include <specularmap_pars_fragment>
#ifdef USE_SPECULARMAP
	
	uniform sampler2D specularMap;
#endif


void main() {


	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;


	//#include <map_fragment>
	#ifdef USE_MAP
		vec4 texelColor = texture2D( map, vUv );

		texelColor = mapTexelToLinear( texelColor );
		diffuseColor *= texelColor;
	#endif


	//#include <specularmap_fragment>
	//USE SPECULAR MAP!
		float specularStrength;
		#ifdef USE_SPECULARMAP

			vec4 texelSpecular = texture2D( specularMap, vUv );
			specularStrength = texelSpecular.r;

		#else

			specularStrength = 1.0;

		#endif

	// accumulation
	//#include <lights_phong_fragment>
	BlinnPhongMaterial material;
	material.diffuseColor = diffuseColor.rgb;
	material.specularColor = specular;
	material.specularShininess = shininess;
	material.specularStrength = specularStrength;

	vec3 normal = normalize( vNormal );

	//#include <lights_template>
	//LIGHTS!!
		/**
		 * This is a template that can be used to light a material, it uses pluggable
		 * RenderEquations (RE)for specific lighting scenarios.
		 *
		 * Instructions for use:
		 * - Ensure that both RE_Direct, RE_IndirectDiffuse and RE_IndirectSpecular are defined
		 * - If you have defined an RE_IndirectSpecular, you need to also provide a Material_LightProbeLOD. <---- ???
		 * - Create a material parameter that is to be passed as the third parameter to your lighting functions.
		 *
		 * TODO:
		 * - Add area light support.
		 * - Add sphere light support.
		 * - Add diffuse light probe (irradiance cubemap) support.
		 */

		GeometricContext geometry;

		geometry.position = - vViewPosition;
		geometry.normal = normal;
		geometry.viewDir = normalize( vViewPosition );

		IncidentLight directLight;

		#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )

			PointLight pointLight;

			for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {

				pointLight = pointLights[ i ];

				getPointDirectLightIrradiance( pointLight, geometry, directLight );

				RE_Direct( directLight, geometry, material, reflectedLight );

			}

		#endif

		#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )

			DirectionalLight directionalLight;

			for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {

				directionalLight = directionalLights[ i ];

				getDirectionalDirectLightIrradiance( directionalLight, geometry, directLight );

				RE_Direct( directLight, geometry, material, reflectedLight );

			}

		#endif

	// modulation

	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;


	gl_FragColor = vec4( outgoingLight, diffuseColor.a );

	//#include <encodings_fragment>
	gl_FragColor = linearToOutputTexel( gl_FragColor );

}
