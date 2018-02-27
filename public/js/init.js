var stats;

var scene;
var camera;

var scene_bg;
var camera_bg;

var renderer;

var camera_root;
var ship_root;

var cameraControls;



var e_screen;
var e_controls;

//-----------------------------
//temp globals - these will be replaced once I implement a real loader
var planet_fs;
var planet_vs;



//-----------------------------

//TO DO -- REPLACE WITH ACTUAL LOADER
function load() {

    // $.get("/shaders/planet_frag.glsl", function(data) {
    //     planet_fs = data;
    //     $.get("/shaders/planet_vert.glsl", function(data) {
    //         planet_vs = data;
    //         init();
    //     });
    // });
}


function init() {

    e_screen = $("#screen");
    e_controls = $("#controls");


    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 100000);

    scene_bg = new THREE.Scene();
    camera_bg = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 100000);



    var light_sun = new THREE.DirectionalLight(0xffffff);
    light_sun.position.set(-0.1, 0.2, 1).normalize();
    light_sun.intensity = 1.0;
    scene.add(light_sun);


    var light_blue = new THREE.DirectionalLight(0x99ccff);
    light_blue.position.set(0.2, -0.5, -1).normalize();
    scene.add(light_blue);

    camera_root = new THREE.Object3D();

    camera_root.add(camera);
    scene.add(camera_root);

    ship_root = new THREE.Object3D();
    scene.add(ship_root);


    //var planet = createPlanet();
    ////planet.scale = new THREE.Vector3(60, 60, 60);
    //scene.add(planet);

    buildBackground(scene_bg);


    renderer = new THREE.WebGLRenderer({
        antialias: true,
    });
    renderer.autoClear = false;

    //Needed for planet shader
    //renderer.context.getExtension('OES_standard_derivatives');

    //append renderer to the screen
    renderer.setSize(e_screen.width(), e_screen.height());
    e_screen.append(renderer.domElement);

    stats = new Stats();
    //stats.showPanel(1);
    e_screen.append(stats.dom);

    window.addEventListener('resize', onWindowResize, false);

    cameraControls = new CameraControls(e_screen, camera_root);

    interfaceControls = new InterfaceControls(e_controls, ship_root);

    //animate();

    var game = {};

    var clock = new Clock(game);

    game.loop = function(_clock) {

        stats.begin();

        cameraControls.update();
        camera_bg.quaternion.copy(camera_root.quaternion);

        //planet.o.update(clock);

        renderer.render(scene_bg, camera_bg);
        renderer.clearDepth();
        renderer.render(scene, camera);


        stats.end();
    }

    clock.start();

}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    camera_bg.aspect = window.innerWidth / window.innerHeight;
    camera_bg.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}