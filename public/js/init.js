var stats;

var scene;
var camera;

var scene_bg;
var camera_bg;

var renderer;

var camera_root;

var cameraControls;


var geometry, material, mesh;



var e_screen;


function init() {

    e_screen = $("#screen");


    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 100000);

    scene_bg = new THREE.Scene();
    camera_bg = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 100000);



    var light_sun = new THREE.DirectionalLight(0xffffff);
    light_sun.position.set(1, 1, 1).normalize();
    scene.add(light_sun);


    var light_blue = new THREE.DirectionalLight(0x99ccff);
    light_blue.position.set(-1, -1, -1).normalize();
    scene.add(light_blue);



    camera_root = new THREE.Object3D();

    camera_root.add(camera);
    scene.add(camera_root);

    //geometry = new THREE.BoxGeometry(200, 200, 200);

    //CylinderGeometry(radiusTop, radiusBottom, height, radiusSegments, heightSegments, openEnded, thetaStart, thetaLength)
    geometry = new THREE.CylinderGeometry(50, 50, 200, 6);
    geometry.computeFlatVertexNormals();
    material = new THREE.MeshPhongMaterial({
        color: 0xffffff,
    });

    mesh = new THREE.Mesh(geometry, material);
    mesh.rotateX(toRad(90));
    scene.add(mesh);



    //--------------------------------------------------------------
    // var geometry_t = new THREE.CylinderGeometry(0, 10, 30, 4, 1);
    // var material_t = new THREE.MeshPhongMaterial({
    //     color: 0xffffff,
    //     flatShading: true
    // });

    // for (var i = 0; i < 500; i++) {
    //     var mesh_t = new THREE.Mesh(geometry_t, material_t);
    //     mesh_t.position.x = (Math.random() - 0.5) * 1000;
    //     mesh_t.position.y = (Math.random() - 0.5) * 1000;
    //     mesh_t.position.z = (Math.random() - 0.5) * 1000;
    //     mesh_t.updateMatrix();
    //     mesh_t.matrixAutoUpdate = false;
    //     scene_bg.add(mesh_t);
    // }
    //--------------------------------------------------------------

    buildBackground(scene_bg);


    renderer = new THREE.WebGLRenderer({
        antialias: true,
    });
    renderer.autoClear = false;

    //append renderer to the screen
    renderer.setSize(e_screen.width(), e_screen.height());
    e_screen.append(renderer.domElement);

    stats = new Stats();
    //stats.showPanel(1);
    e_screen.append(stats.dom);

    window.addEventListener('resize', onWindowResize, false);

    cameraControls = new CameraControls(e_screen, camera_root);

    animate();
}


function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    camera_bg.aspect = window.innerWidth / window.innerHeight;
    camera_bg.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}





function animate() {

    stats.begin();

    requestAnimationFrame(animate);

    cameraControls.update();
    camera_bg.quaternion.copy(camera_root.quaternion);

    renderer.render(scene_bg, camera_bg);
    renderer.clearDepth();
    renderer.render(scene, camera);


    stats.end();
}