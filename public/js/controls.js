function CameraControls(e_screen, camera_root) {
    var mouseIsDown = false;
    var spindown = false;
    var oldMousePos = new THREE.Vector2(0, 0);
    var mouseDelta = new THREE.Vector2(0, 0);
    var rotationDelta = new THREE.Vector2(0, 0);

    e_screen.on("mousedown", function(event) {
        mouseIsDown = true;
        spindown = false;
        oldMousePos.set(event.clientX, event.clientY);
        //console.log("Down");
    });

    e_screen.on("mouseup", function(event) {
        mouseIsDown = false;
        spindown = true;
        //console.log("Up");
    });

    e_screen.on("mousemove", function(event) {
        if (mouseIsDown) {
            mouseDelta.subVectors(new THREE.Vector2(event.clientX, event.clientY), oldMousePos);
            oldMousePos.set(event.clientX, event.clientY);
            rotationDelta.set(-mouseDelta.y, -mouseDelta.x);
        }
    });



    this.update = function() {
        if (mouseIsDown) {

            camera_root.rotateX(toRad(rotationDelta.x));
            camera_root.rotateY(toRad(rotationDelta.y));
            rotationDelta.set(0, 0);

        } else if (spindown) {
            rotationDelta.multiplyScalar(0.85);
            if (rotationDelta.lengthManhattan() < 0.000000001) {
                rotationDelta.set(0, 0);
                spindown = false;
            }
            camera_root.rotateX(toRad(rotationDelta.x));
            camera_root.rotateY(toRad(rotationDelta.y));
        }
    }
}