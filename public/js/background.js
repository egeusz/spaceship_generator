function buildBackground(scene) {

    // var light_sun = new THREE.DirectionalLight(0xffffff);
    // light_sun.position.set(1, 1, 1).normalize();
    // scene.add(light_sun);

    buildStars(scene);

}


function buildStars(scene) {

    var geometry = new THREE.Geometry();
    var sprite = new THREE.TextureLoader().load("/images/textures/stars/star1.png");
    for (i = 0; i < 1000; i++) {
        var vertex = new THREE.Vector3();
        vertex.x = 2000 * Math.random() - 1000;
        vertex.y = 2000 * Math.random() - 1000;
        vertex.z = 2000 * Math.random() - 1000;

        if (vertex.lengthSq() > 200 * 200) {
            geometry.vertices.push(vertex);
        }


    }
    var material = new THREE.PointsMaterial({
        size: 10,
        sizeAttenuation: true,
        map: sprite,
        alphaTest: 0.5,
        transparent: true
    });
    material.blending = THREE.AdditiveBlending;
    //material.color.setHSL(1.0, 0.3, 0.7);
    var particles = new THREE.Points(geometry, material);

    scene.add(particles);
}