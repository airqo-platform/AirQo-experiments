const express = require('express')
const app = express();

app.get('/', (req, res) => {
  res.send('Testing nodejs Microservice!!-Version 1.45')
});

app.listen(8000, () => {
  console.log('Example app listening on port 8000!')
});