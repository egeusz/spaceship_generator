var createPlanet = function() {
    var planet = new THREE.Object3D();

    var texture_url = "/images/textures/planet/";

    var texture_diff = new THREE.TextureLoader().load(texture_url + "carpathia_diff.png");
    var texture_spec = new THREE.TextureLoader().load(texture_url + "carpathia_spec.png");
    var texture_clouds = new THREE.TextureLoader().load(texture_url + "earth_clouds.png");
    var texture_cloud_disp = new THREE.TextureLoader().load(texture_url + "cloud_disp.png");

    var texture_glow_night = new THREE.TextureLoader().load(texture_url + "carpathia_glow_night.png");
    var texture_glow_day = new THREE.TextureLoader().load(texture_url + "carpathia_glow_day.png");
    var texture_glow_scatter = new THREE.TextureLoader().load(texture_url + "carpathia_glow_scatter.png");


    //------- Object Parameters
    //Doing this to keep my sloppy ass from overriding any THREE stuff in THREE.Object3D
    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    // planet.o = {};
    // planet.o.id 					  = _id;
    // planet.o.rotationSpeed            = _data.rotation;
    // planet.o.dispSpeed 	              = _data.wind*0.25;
    // planet.o.cloudDisplacementOffset  = 0.0;

    // //info
    // planet.o.name 					  =_data.name;
    // planet.o.description 			  =_data.description;



    // planet.o.size 			          =_data.size;
    // planet.o.screenPosition 		  = new THREE.Vector2();
    // planet.o.distance 		          = 0;
    // planet.o.screenSize 		      = 0;

    // planet.o.spawner                  =_spawner;

    // if(_assetList.text[_id+"_details"])
    // {
    // 	planet.o.details   =   _assetList.text[_id+"_details"];
    // }


    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    //-------Build Material
    //var material = new THREE.MeshBasicMaterial( { map: _assetList.texture[_id+'_diff'] } );
    /*
    var material = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.merge([
            THREE.UniformsLib['lights'],
            THREE.UniformsLib['ambient'], {
                atmoColor: {
                    type: 'c',
                    value: 0.0
                },
                hoverColor: {
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
                txtrglowday: {
                    type: 't',
                    value: null
                },
                txtrglownight: {
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
        //vertexShader: document.getElementById('planet_vertexShader').text,
        //fragmentShader: document.getElementById('planet_fragmentShader').text,
        vertexShader: planet_vs,
        fragmentShader: planet_fs,
        //vertexShader: _assetList.shader['planet_vert'],
        //fragmentShader: _assetList.shader['planet_frag'],
        transparent: false,
        lights: true
    });
    //-------Assign Texturez
    material.uniforms.txtrdiff.value = texture_diff;
    material.uniforms.txtrspec.value = texture_spec;
    material.uniforms.txtrclouds.value = texture_clouds;
    material.uniforms.txtrclouddisp.value = texture_cloud_disp;

    material.uniforms.txtrglownight.value = texture_glow_night;
    material.uniforms.txtrglowday.value = texture_glow_day;
    material.uniforms.txtrglowscatter.value = texture_glow_scatter;

    material.uniforms.atmoColor.value = new THREE.Color('#65adff');
    material.uniforms.hoverColor.value = new THREE.Color('#000000');
    material.uniforms.clouddispscale.value = 0;
	*/

    //-------------------------------------------
    //material = new THREE.MeshPhongMaterial();
    material = new THREE.ShaderMaterial({
        uniforms: THREE.UniformsUtils.merge([
            THREE.UniformsLib['lights'],
            THREE.UniformsLib['ambient'],
        ]),
        //vertexShader: document.getElementById('planet_vertexShader').text,
        //fragmentShader: document.getElementById('planet_fragmentShader').text,
        vertexShader: planet_vs,
        fragmentShader: planet_fs,
        //vertexShader: _assetList.shader['planet_vert'],
        //fragmentShader: _assetList.shader['planet_frag'],
        transparent: false,
        lights: true
    });


    //material = new THREE.MeshPhongMaterial();
    material.color = new THREE.Color('#65adff');

    //------Build Mesh
    //var mesh = new THREE.Mesh(_assetList.obj.planet_obj.children[0].children[2].geometry, material);

    var geometry = new THREE.SphereGeometry(1, 64, 64);



    var mesh = new THREE.Mesh(geometry, material);

    mesh.scale.x = 100;
    mesh.scale.y = 100;
    mesh.scale.z = 100;

    planet.add(mesh);

    planet.position.x = 0;
    planet.position.y = 0;
    planet.position.z = -3;

    //console.log(THREE.ShaderChunk);

    // //------Functions
    // //Setup
    // planet.o.updateSceenPosition = function(_sceneRaycaster) {

    //     planet.o.screenPosition = _sceneRaycaster.getScreenLocation(planet.position);
    //     planet.o.distance = _sceneRaycaster.getDistanceTo(planet.position);
    //     planet.o.screenSize = _sceneRaycaster.getScreenWidth(planet.position, planet.o.distance, planet.o.size);

    // }

    // planet.o.update = function(_clock) {
    //     planet.rotation.y += _clock.timeScale(planet.o.rotationSpeed);



    //     //Update Clouds
    //     planet.o.cloudDisplacementOffset += _clock.timeScale(planet.o.dispSpeed);
    //     if (planet.o.cloudDisplacementOffset >= 1.0) {
    //         planet.o.cloudDisplacementOffset -= 1.0;
    //     } else if (planet.o.cloudDisplacementOffset <= 0) {
    //         planet.o.cloudDisplacementOffset += 1.0;
    //     }
    //     planet.o.material.uniforms.clouddispscale.value = planet.o.cloudDisplacementOffset;
    // }

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