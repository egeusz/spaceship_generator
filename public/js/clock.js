//////////////////////////////////////////////////////////
// By Eric Geusz 
// © 2014
//////////////////////////////////////////////////////////

function Clock(_game) {
    var oldTime = Date.now();
    var newTime = Date.now();
    var deltatime = 0;
    var deltatimeSeconds = 0;

    var update = function() {

        oldTime = newTime;
        newTime = Date.now();
        deltatime = newTime - oldTime;
        deltatimeSeconds = deltatime * 0.0001;
        //console.log(deltatimeSeconds);
    }

    var tick = function() {

        requestAnimationFrame(tick);
        update();
        _game.loop(this);


    }

    this.start = function() {
        tick();
    }

    this.timeScale = function(_value) {
        return _value * deltatimeSeconds;
    }




}