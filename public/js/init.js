var scene, camera, renderer;
var geometry, material, mesh;





function init() {



    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 10000);
    camera.position.z = 1000;



    var light_sun = new THREE.DirectionalLight(0xffffff);
    light_sun.position.set(1, 1, 1).normalize();
    scene.add(light_sun);

    var light_blue = new THREE.DirectionalLight(0x99ccff);
    light_blue.position.set(-1, -1, -1).normalize();
    scene.add(light_blue);


    //geometry = new THREE.BoxGeometry(200, 200, 200);

    //CylinderGeometry(radiusTop, radiusBottom, height, radiusSegments, heightSegments, openEnded, thetaStart, thetaLength)
    geometry = new THREE.CylinderGeometry(50, 50, 200, 6);
    geometry.computeFlatVertexNormals();
    material = new THREE.MeshPhongMaterial({
        color: 0xffffff,
    });

    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);



    renderer = new THREE.WebGLRenderer({
        antialias: true
    });

    setUpUI(renderer)
    setUpControls(camera);
}

function animate() {

    requestAnimationFrame(animate);

    mesh.rotation.x += 0.01;
    mesh.rotation.y += 0.02;

    renderer.render(scene, camera);

}



init();
animate();