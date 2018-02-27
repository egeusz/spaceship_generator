var createPlanet = function() {
    var planet = new THREE.Object3D();

    var texture_url = "/images/textures/planet/";

    var texture_diff = new THREE.TextureLoader().load(texture_url + "earth_diff_xl.jpg");
    var texture_spec = new THREE.TextureLoader().load(texture_url + "earth_spec_xl.jpg");
    var texture_bump = new THREE.TextureLoader().load(texture_url + "earth_bump_xl.jpg");

    var texture_cloud_bump = new THREE.TextureLoader().load(texture_url + "cloud_bump.jpg");

    var texture_clouds = new THREE.TextureLoader().load(texture_url + "earth_clouds2_xl.png");
    var texture_clouds_alpha_noise = new THREE.TextureLoader().load(texture_url + "cloud_alpha_noise.jpg");
    var texture_cloud_disp = new THREE.TextureLoader().load(texture_url + "cloud_disp3.png");

    var texture_lights = new THREE.TextureLoader().load(texture_url + "earth_glow_xl.jpg");
    var texture_lights_scatter = new THREE.TextureLoader().load(texture_url + "earth_glow_scatter.jpg");
    var texture_lights_offset = new THREE.TextureLoader().load(texture_url + "lightswitch_offset2.jpg");

    //-------------------------------------------
    material = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.merge([
            THREE.UniformsLib['lights'],
            THREE.UniformsLib['ambient'], {
                //------------------
                map_diff: {
                    type: 't',
                    value: null
                },
                map_spec: {
                    type: 't',
                    value: null
                },
                //------------------
                map_surface_bump: {
                    type: 't',
                    value: null
                },
                surface_bump_scale: {
                    type: 'f',
                    value: 0.0
                },
                //------------------
                map_clouds: {
                    type: 't',
                    value: null
                },
                map_cloud_noise: {
                    type: 't',
                    value: null
                },
                map_cloud_disp: {
                    type: 't',
                    value: null
                },
                cloud_height: {
                    type: 'f',
                    value: 10.0
                },
                cloud_disp: {
                    type: 'f',
                    value: 0.0
                },
                color_cloud_shadow: {
                    type: 'c',
                    value: 0.0
                },
                //------------------
                map_cloud_bump: {
                    type: 't',
                    value: null
                },
                cloud_bump_scale: {
                    type: 'f',
                    value: 0.0
                },
                //------------------
                color_atmo: {
                    type: 'c',
                    value: 0.0
                },
                color_atmoscatter_sunset: {
                    type: 'c',
                    value: 0.0
                },
                color_atmoscatter_night: {
                    type: 'c',
                    value: 0.0
                },
                //------------------
                map_lights: {
                    type: 't',
                    value: null
                },
                map_lights_scatter: {
                    type: 't',
                    value: null
                },
                map_lights_offset: {
                    type: 't',
                    value: null
                },
                color_lights_brightness: {
                    type: 'c',
                    value: 0.0
                },
            }
        ]),
        vertexShader: planet_vs,
        fragmentShader: planet_fs,

        transparent: false,
        lights: true
    });



    //-------Assign Texturez
    material.uniforms.map_diff.value = texture_diff;
    material.uniforms.map_diff.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_diff.value.wrapT = THREE.RepeatWrapping;

    material.uniforms.map_spec.value = texture_spec;
    material.uniforms.map_spec.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_spec.value.wrapT = THREE.RepeatWrapping;

    material.uniforms.map_surface_bump.value = texture_bump;
    material.uniforms.map_surface_bump.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_surface_bump.value.wrapT = THREE.RepeatWrapping;
    material.uniforms.surface_bump_scale.value = 0.3;

    material.uniforms.map_clouds.value = texture_clouds;
    material.uniforms.map_clouds.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_clouds.value.wrapT = THREE.RepeatWrapping;

    material.uniforms.map_cloud_noise.value = texture_clouds_alpha_noise;
    material.uniforms.map_cloud_noise.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_cloud_noise.value.wrapT = THREE.RepeatWrapping;

    material.uniforms.map_cloud_disp.value = texture_cloud_disp;
    material.uniforms.map_cloud_disp.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_cloud_disp.value.wrapT = THREE.RepeatWrapping;

    material.uniforms.cloud_height.value = 15;
    material.uniforms.cloud_disp.value = 0;
    material.uniforms.color_cloud_shadow.value = new THREE.Color('#557190');

    material.uniforms.map_cloud_bump.value = texture_cloud_bump;
    material.uniforms.map_cloud_bump.value.wrapS = THREE.RepeatWrapping;
    material.uniforms.map_cloud_bump.value.wrapT = THREE.RepeatWrapping;
    material.uniforms.cloud_bump_scale.value = 0.1;

    material.uniforms.color_atmo.value = new THREE.Color('#65adff');
    material.uniforms.color_atmoscatter_sunset.value = new THREE.Color('#ff2400');
    material.uniforms.color_atmoscatter_night.value = new THREE.Color('#0066ff');

    material.uniforms.map_lights.value = texture_lights;
    material.uniforms.map_lights_scatter.value = texture_lights_scatter;
    material.uniforms.map_lights_offset.value = texture_lights_offset;
    material.uniforms.color_lights_brightness.value = new THREE.Color('#ffffff');

    //material.uniforms.map_glow.value = texture_glow;
    //material.uniforms.map_glowscatter.value = texture_glow_scatter;

    //------Build Mesh

    var geometry = new THREE.SphereGeometry(1, 128, 128);

    var mesh = new THREE.Mesh(geometry, material);

    mesh.scale.x = 300;
    mesh.scale.y = 300;
    mesh.scale.z = 300;

    planet.add(mesh);

    planet.position.x = 0;
    planet.position.y = 0;
    planet.position.z = 0;

    //------Functions
    planet.o = {
        material: material,
        cloudDisplacementOffset: 0,
    };

    planet.o.update = function(_clock) {
        planet.rotation.y += _clock.timeScale(0.5);

        //Update Clouds
        planet.o.cloudDisplacementOffset += _clock.timeScale(0.02);
        planet.o.cloudDisplacementOffset = planet.o.cloudDisplacementOffset % 1;

        planet.o.material.uniforms.cloud_disp.value = planet.o.cloudDisplacementOffset;
    }

    // //Display
    // planet.o.setStyleOnHovered = function() {
    //     planet.o.material.uniforms.hoverColor.value = new THREE.Color('#151525');
    // }
    // planet.o.setStyleOffHovered = function() {
    //     planet.o.material.uniforms.hoverColor.value = new THREE.Color('#000000');
    // }


    // //Interaction
    // planet.o.onMouseHover = function(_mousePosition, _distance) {
    //     planet.o.spawner.onMouseHover(planet, _mousePosition, _distance)
    // }
    // planet.o.offMouseHover = function() {
    //     planet.o.spawner.offMouseHover(planet);
    // }
    // planet.o.onMouseClick = function(_mousePosition, _distance) {
    //     planet.o.spawner.onPlanetClicked(planet, _mousePosition, _distance)
    // }



    return planet;
}