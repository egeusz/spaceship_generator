function CameraControls(e_screen, camera_root) {
    var self = this;
    var camera = camera_root.children[0];

    var mouseIsDown = false;
    var spindown = false;

    var oldMousePos = new THREE.Vector2(0, 0);
    var mouseDelta = new THREE.Vector2(0, 0);
    var rotationDelta = new THREE.Vector2(0, 0);

    var isZooming = false;

    var zoomPos = 1000;
    //var zoomDelta = 0;
    var zoomTarget = zoomPos;
    camera.position.setZ(zoomPos);

    this.zoomMax = 7000;
    this.zoomMin = 375;
    //this.zoomMin_stopbuffer = 500;
    this.zoomSpeed = 150;

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

    e_screen.bind('mousewheel DOMMouseScroll', function(event) {

        if (event.originalEvent.wheelDelta > 0 || event.originalEvent.detail < 0) {
            // scroll up
            //console.log("Zoom Out");
            isZooming = true;
            zoomTarget += self.zoomSpeed;
            if (zoomTarget > self.zoomMax) {
                zoomTarget = self.zoomMax;
            }

        } else {
            // scroll down
            //console.log("Zoom In");
            isZooming = true;
            zoomTarget -= self.zoomSpeed;
            if (zoomTarget < self.zoomMin) {
                zoomTarget = self.zoomMin;
            }
        }
    });


    this.update = function() {
        //Rotation
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

        //Zoom
        if (isZooming) {

            var zoomDelta = (zoomTarget - zoomPos) * 0.10;
            if (Math.abs(zoomDelta) < 0.000000001) {
                zoomDelta = 0;
                isZooming = false;
            }
            zoomPos += zoomDelta;

            camera.position.setZ(zoomPos);
        }

    }
}