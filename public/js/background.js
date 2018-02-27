function buildBackground(scene, fg_scene) {

    var light_sun = new THREE.DirectionalLight(0xffffff);
    light_sun.position.set(1, 1, 1).normalize();
    scene.add(light_sun);

    buildStars(scene);

    // var planet = createPlanet();
    // scene.add(planet);

}


function buildStars(scene) {

    var sprite_star_small = new THREE.TextureLoader().load("/images/textures/stars/star_small.png");
    var sprite_star_blue = new THREE.TextureLoader().load("/images/textures/stars/star1.png");
    //------------------------------------------------------------
    { //small stars

        var geometry = new THREE.Geometry();
        for (i = 0; i < 2000; i++) {
            var vertex = new THREE.Vector3();
            vertex.x = 4000 * Math.random() - 2000;
            vertex.y = 4000 * Math.random() - 2000;
            vertex.z = 4000 * Math.random() - 2000;

            //ignore any that are too close to the camera
            if (vertex.lengthSq() > 500 * 500) {
                //i--;
                geometry.vertices.push(vertex);
            } else {
                i--;
            }


        }
        var material = new THREE.PointsMaterial({
            size: 5,
            sizeAttenuation: true,
            map: sprite_star_small,
            alphaTest: 0.5,
            transparent: true
        });
        material.blending = THREE.AdditiveBlending;
        //material.color.setHex(0x666666);
        //material.color.setHSL(1.0, 0.3, 0.7);
        var particles = new THREE.Points(geometry, material);

        scene.add(particles);
    }
    //------------------------------------------------------------
    { //blue stars

        var geometry = new THREE.Geometry();
        for (i = 0; i < 400; i++) {
            var vertex = new THREE.Vector3();
            vertex.x = 1000 * Math.random() - 500;
            vertex.y = 1000 * Math.random() - 500;
            vertex.z = 1000 * Math.random() - 500;

            //ignore any that are too close to the camera
            if (vertex.lengthSq() > 50 * 50) {
                //i--;
                geometry.vertices.push(vertex);
            } else {
                i--;
            }


        }
        var material = new THREE.PointsMaterial({
            size: 6,
            sizeAttenuation: true,
            map: sprite_star_blue,
            alphaTest: 0.5,
            transparent: true
        });
        material.blending = THREE.AdditiveBlending;
        var particles = new THREE.Points(geometry, material);

        scene.add(particles);
    }
    //------------------------------------------------------------
    {








    }
}