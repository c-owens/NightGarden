// A tiny static server for Night Garden: `node serve.js`, then open http://localhost:8777/
// No dependencies. Serves the index at "/" and every scene and music file beside it.
// Set PORT to use a different port.
const http = require('http'), fs = require('fs'), path = require('path');

const root = __dirname;
const port = Number(process.env.PORT) || 8777;
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.png': 'image/png', '.svg': 'image/svg+xml',
};

http.createServer((req, res) => {
  let url;
  try { url = decodeURIComponent(req.url.split('?')[0]); } catch { res.writeHead(400); return res.end(); }
  const file = path.join(root, url === '/' ? 'index.html' : url);
  if (file !== root && !file.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': types[path.extname(file).toLowerCase()] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(port, () => console.log(`Night Garden: http://localhost:${port}/`));
