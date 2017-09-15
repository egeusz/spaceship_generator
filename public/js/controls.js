var e_screen;

function setUpUI(renderer) {
    //find the screen
    e_screen = $("#screen");


    //append renderer to the screen
    renderer.setSize(e_screen.width(), e_screen.height());
    e_screen.append(renderer.domElement);
}



function setUpControls(camera) {

}