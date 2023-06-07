import express from 'express';
import path from 'path';

const app = express();
app.use(express.static(__dirname + '/dist'));
app.get('/*', (_, res) => res.sendFile(path.join(__dirname + '/dist/index.html')));
app.listen(process.env.PORT || 8080);
