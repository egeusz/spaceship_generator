var createPlanet = function() {
    var planet = new THREE.Object3D();

    var texture_url = "/images/textures/planet/";

    var texture_diff = new THREE.TextureLoader().load(texture_url + "earth_diff_lg.jpg");
    var texture_spec = new THREE.TextureLoader().load(texture_url + "earth_spec_lg.jpg");

    var texture_clouds = new THREE.TextureLoader().load(texture_url + "earth_clouds_lg.png");
    var texture_cloud_disp = new THREE.TextureLoader().load(texture_url + "cloud_disp.png");

    var texture_lights = new THREE.TextureLoader().load(texture_url + "earth_glow_lg.jpg");
    var texture_lights_scatter = new THREE.TextureLoader().load(texture_url + "earth_glow_scatter.jpg");
    var texture_lights_offset = new THREE.TextureLoader().load(texture_url + "lightswitch_offset.jpg");

    var texture_glow = new THREE.TextureLoader().load(texture_url + "carpathia_glow_day.png");
    var texture_glow_scatter = new THREE.TextureLoader().load(texture_url + "carpathia_glow_scatter.png");

    //-------------------------------------------
    material = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.merge([
            THREE.UniformsLib['lights'],
            THREE.UniformsLib['ambient'], {
                coloratmo: {
                    type: 'c',
                    value: 0.0
                },
                coloratmoscatter1: {
                    type: 'c',
                    value: 0.0
                },
                coloratmoscatter2: {
                    type: 'c',
                    value: 0.0
                },
                hovercolor: {
                    type: 'c',
                    value: 0.0
                },

                txtrdiff: {
                    type: 't',
                    value: null
                },
                txtrspec: {
                    type: 't',
                    value: null
                },
                txtrclouds: {
                    type: 't',
                    value: null
                },
                txtrclouddisp: {
                    type: 't',
                    value: null
                },
                txtrlights: {
                    type: 't',
                    value: null
                },
                txtrlightsscatter: {
                    type: 't',
                    value: null
                },
                txtrlightsoffset: {
                    type: 't',
                    value: null
                },
                txtrglow: {
                    type: 't',
                    value: null
                },
                txtrglowscatter: {
                    type: 't',
                    value: null
                },

                clouddispscale: {
                    type: 'f',
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
    material.uniforms.txtrdiff.value = texture_diff;
    material.uniforms.txtrspec.value = texture_spec;
    material.uniforms.txtrclouds.value = texture_clouds;
    material.uniforms.txtrclouddisp.value = texture_cloud_disp;

    material.uniforms.txtrlights.value = texture_lights;
    material.uniforms.txtrlightsscatter.value = texture_lights_scatter;
    material.uniforms.txtrlightsoffset.value = texture_lights_offset;

    material.uniforms.txtrglow.value = texture_glow;
    material.uniforms.txtrglowscatter.value = texture_glow_scatter;

    material.uniforms.coloratmo.value = new THREE.Color('#65adff');
    material.uniforms.coloratmoscatter1.value = new THREE.Color('#ff2400');
    material.uniforms.coloratmoscatter2.value = new THREE.Color('#0066ff');
    material.uniforms.hovercolor.value = new THREE.Color('#000000');
    material.uniforms.clouddispscale.value = 0;

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

    // //Setup4
    // planet.o.updateSceenPosition = function(_sceneRaycaster) {

    //     planet.o.screenPosition = _sceneRaycaster.getScreenLocation(planet.position);
    //     planet.o.distance = _sceneRaycaster.getDistanceTo(planet.position);
    //     planet.o.screenSize = _sceneRaycaster.getScreenWidth(planet.position, planet.o.distance, planet.o.size);

    // }

    planet.o.update = function(_clock) {
        planet.rotation.y += _clock.timeScale(0.25);

        //Update Clouds
        planet.o.cloudDisplacementOffset += _clock.timeScale(0.15);
        if (planet.o.cloudDisplacementOffset >= 1.0) {
            planet.o.cloudDisplacementOffset -= 1.0;
        } else if (planet.o.cloudDisplacementOffset <= 0) {
            planet.o.cloudDisplacementOffset += 1.0;
        }
        planet.o.material.uniforms.clouddispscale.value = planet.o.cloudDisplacementOffset;
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