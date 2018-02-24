var shipspec = {
    segments: [{
        length: 100,
        radius: 50,
        symmetry: 0,
        sides: 24,

        spokes: [],

    }, {
        length: 200,
        radius: 50,
        symmetry: 3,
        sides: 3,

        spokes: [],

    }, {
        length: 150,
        radius: 50,
        symmetry: 4,
        sides: 4,

        spokes: [],

    }, {
        length: 175,
        radius: 50,
        symmetry: 6,
        sides: 6,

        spokes: [{
            type: "cube",
            length: 50,
            offset: 25,
            radius: 25,
            symmetry: 3,
            onFace: true,
        }]
    }, {
        length: 100,
        radius: 50,
        symmetry: 8,
        sides: 8,

        spokes: [],
    }]
}