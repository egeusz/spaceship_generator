const express = require('express');
const path = require('path');
const app = express();

// app.get('/', function (req, res) {
//   res.send('Hello World!')
// })

//app.use('/', express.static('../public'));


app.use(express.static(path.join(__dirname, '..', 'public')));


app.listen(3000, function() {
    console.log('Example app listening on port 3000!')
});