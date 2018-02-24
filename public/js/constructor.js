function constructShip(spec, ship_root) {

    console.log("Constructing Ship ...");




    var positionSum = 0;

    for (var i = 0; i < spec.segments.length; i++) {

        seg = spec.segments[i];
        positionSum += seg.length / 2;

        //------------------------------------------
        //build segment
        var geometry = new THREE.CylinderGeometry(seg.radius, seg.radius, seg.length, seg.sides);
        geometry.computeFlatVertexNormals();
        var material = new THREE.MeshPhongMaterial({
            color: 0xffffff,
        });

        var segment_mesh = new THREE.Mesh(geometry, material);

        //------------------------------------------
        //build spokes
        if (seg.spokes) {
            console.log("HI");
            // for (var i = 0; i < seg.spokes.length; i++) {
            //     spoke = seg.spokes[i];

            //     makeSpoke = ContructionMethods.spokes[spoke.type];
            //     if (makeSpoke) {
            //         makeSpoke(segment_mesh, spoke);
            //     }
            // }

        }

        //------------------------------------------
        //build rings
        if (seg.rings) {


        }



        //mesh.rotateX(toRad(90));
        segment_mesh.position.y = positionSum;
        positionSum += seg.length / 2;

        ship_root.add(segment_mesh);
    }

    //recenter the ship. 
    ship_root.position.y = -(positionSum / 2);





    return ship_root;

}


var ContructionMethods = {

    spokes: {

        cube: function(seg, spec_spoke) {

            var geometry = new THREE.Cube(100, 100, 100);
            geometry.computeFlatVertexNormals();
            var material = new THREE.MeshPhongMaterial({
                color: 0xffffff,
            });

            var mesh = new THREE.Mesh(geometry, material);
            mesh.position.x = 200;

            seg.add(mesh);

            return seg;

        }
    }

}