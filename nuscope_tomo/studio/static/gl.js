// NUScope-TOMO Studio — WebGL2 ray-marching volume renderer.
// Modes: 0 = direct volume rendering, 1 = MIP, 2 = shaded iso-surface,
// 3 = label surfaces (with grey values on clip faces).
"use strict";

const VS = `#version 300 es
in vec3 pos;
uniform mat4 mvp;
uniform vec3 boxSize;
out vec3 vPos;
void main(){ vPos = pos * boxSize; gl_Position = mvp * vec4(vPos, 1.0); }`;

const FS = `#version 300 es
precision highp float; precision highp sampler3D;
in vec3 vPos;
out vec4 outColor;
uniform sampler3D vol;
uniform sampler3D lab;
uniform sampler2D cmap;
uniform sampler2D lcol;
uniform vec3 boxSize, camPos, clipMin, clipMax, texSize;
uniform int mode, hasVol, hasLab, cutFaces;
uniform float thr, dens, wlo, whi;

vec2 hitBox(vec3 o, vec3 d, vec3 bmin, vec3 bmax){
  vec3 inv = 1.0 / d;
  vec3 t0 = (bmin - o) * inv, t1 = (bmax - o) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
float sv(vec3 p){ float v = texture(vol, p).r; return clamp((v - wlo) / max(whi - wlo, 1e-3), 0.0, 1.0); }
float sl(vec3 p){ return texture(lab, p).r * 255.0; }
vec3 grad(vec3 p){
  vec3 e = 1.0 / texSize;
  return vec3(sv(p+vec3(e.x,0,0))-sv(p-vec3(e.x,0,0)), sv(p+vec3(0,e.y,0))-sv(p-vec3(0,e.y,0)),
              sv(p+vec3(0,0,e.z))-sv(p-vec3(0,0,e.z)));
}
vec3 lgrad(vec3 p, float id){
  vec3 e = 1.5 / texSize;
  return vec3(float(abs(sl(p+vec3(e.x,0,0))-id)<.5)-float(abs(sl(p-vec3(e.x,0,0))-id)<.5),
              float(abs(sl(p+vec3(0,e.y,0))-id)<.5)-float(abs(sl(p-vec3(0,e.y,0))-id)<.5),
              float(abs(sl(p+vec3(0,0,e.z))-id)<.5)-float(abs(sl(p-vec3(0,0,e.z))-id)<.5));
}
vec3 shade(vec3 base, vec3 n, vec3 v){
  n = normalize(n + 1e-6);
  if (dot(n, v) < 0.0) n = -n;
  vec3 L1 = normalize(v + vec3(0.3, 0.5, 0.2)), L2 = normalize(vec3(-0.5, -0.3, 0.6));
  float d = 0.25 + 0.65 * max(dot(n, L1), 0.0) + 0.2 * max(dot(n, L2), 0.0);
  float s = pow(max(dot(reflect(-L1, n), v), 0.0), 24.0) * 0.35;
  return base * d + s;
}
void main(){
  vec3 o = camPos / boxSize + 0.5;
  vec3 d = normalize((vPos - camPos) / boxSize);
  vec2 t = hitBox(o, d, clipMin, clipMax);
  if (t.x > t.y) discard;
  t.x = max(t.x, 0.0);
  vec3 dd = d * boxSize; float worldLen = length(dd);
  float steps = 1.6 * max(max(texSize.x, texSize.y), texSize.z);
  float dt = 1.0 / steps;
  vec3 p = o + d * t.x;
  vec3 view = -normalize(d);
  // entry face of the clip box: is it a cut (not an original box face)?
  bool cut = false;
  for (int k = 0; k < 3; k++) {
    if ((abs(p[k] - clipMin[k]) < 1e-3 && clipMin[k] > 1e-3) ||
        (abs(p[k] - clipMax[k]) < 1e-3 && clipMax[k] < 0.999)) cut = true;
  }
  if (mode == 3) {
    if (cutFaces == 1 && cut && hasVol == 1) {
      float id = hasLab == 1 ? sl(p) : 0.0;
      vec3 g = vec3(sv(p));
      vec4 lc = texture(lcol, vec2((id + 0.5) / 256.0, 0.5));
      outColor = vec4(mix(g, lc.rgb, id > 0.5 ? 0.55 * lc.a : 0.0), 1.0); return;
    }
    for (float s = t.x; s < t.y; s += dt) {
      vec3 q = o + d * s;
      float id = sl(q);
      if (id > 0.5) {
        vec4 lc = texture(lcol, vec2((id + 0.5) / 256.0, 0.5));
        if (lc.a > 0.01) { outColor = vec4(shade(lc.rgb, -lgrad(q, id) / boxSize, view), 1.0); return; }
      }
    }
    discard;
  }
  if (mode == 2) {
    if (cutFaces == 1 && cut && sv(p) > thr) { outColor = vec4(texture(cmap, vec2(sv(p), .5)).rgb, 1); return; }
    for (float s = t.x; s < t.y; s += dt) {
      vec3 q = o + d * s;
      if (sv(q) > thr) { vec3 c = texture(cmap, vec2(0.75, .5)).rgb;
        outColor = vec4(shade(c, -grad(q) / boxSize, view), 1.0); return; }
    }
    discard;
  }
  if (mode == 1) {
    float m = 0.0;
    for (float s = t.x; s < t.y; s += dt) m = max(m, sv(o + d * s));
    outColor = vec4(texture(cmap, vec2(m, .5)).rgb, 1.0); return;
  }
  vec4 acc = vec4(0);
  float stepLen = dt * worldLen * 100.0;
  for (float s = t.x; s < t.y; s += dt) {
    vec3 q = o + d * s;
    float v = sv(q);
    float a = smoothstep(thr, min(thr + 0.35, 1.0), v) * dens * 0.08 * stepLen;
    if (a > 0.002) {
      vec3 c = texture(cmap, vec2(v, .5)).rgb;
      c = shade(c, -grad(q) / boxSize, view) * 0.85 + c * 0.15;
      if (hasLab == 1) { float id = sl(q); vec4 lc = texture(lcol, vec2((id + .5) / 256., .5));
        if (id > 0.5 && lc.a > 0.01) { c = mix(c, lc.rgb, 0.7); a = max(a, 0.12 * stepLen); } }
      acc.rgb += (1.0 - acc.a) * a * c; acc.a += (1.0 - acc.a) * a;
      if (acc.a > 0.97) break;
    }
  }
  outColor = vec4(acc.rgb, acc.a);
}`;

const CMAPS = {
  "bone": [[0,0,0],[84,84,116],[167,199,199],[255,255,255]],
  "gray": [[0,0,0],[255,255,255]],
  "inferno": [[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]],
  "viridis": [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
  "ice-fire": [[0,0,0],[27,61,133],[76,194,255],[255,255,255],[255,170,40],[230,60,20]],
};

class VolumeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: true });
    if (!gl) throw new Error("WebGL2 not available");
    this.prog = this._program(VS, FS);
    const cube = new Float32Array([
      -.5,-.5,-.5, .5,-.5,-.5, .5,.5,-.5, -.5,-.5,-.5, .5,.5,-.5, -.5,.5,-.5,
      -.5,-.5,.5, .5,.5,.5, .5,-.5,.5, -.5,-.5,.5, -.5,.5,.5, .5,.5,.5,
      -.5,-.5,-.5, -.5,.5,.5, -.5,-.5,.5, -.5,-.5,-.5, -.5,.5,-.5, -.5,.5,.5,
      .5,-.5,-.5, .5,-.5,.5, .5,.5,.5, .5,-.5,-.5, .5,.5,.5, .5,.5,-.5,
      -.5,-.5,-.5, -.5,-.5,.5, .5,-.5,.5, -.5,-.5,-.5, .5,-.5,.5, .5,-.5,-.5,
      -.5,.5,-.5, .5,.5,.5, -.5,.5,.5, -.5,.5,-.5, .5,.5,-.5, .5,.5,.5]);
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, cube, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.prog, "pos");
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    this.u = {};
    for (const n of ["mvp","boxSize","camPos","clipMin","clipMax","texSize","mode","hasVol","hasLab",
                     "cutFaces","thr","dens","wlo","whi","vol","lab","cmap","lcol"])
      this.u[n] = gl.getUniformLocation(this.prog, n);
    this.volTex = this._tex3d(new Uint8Array([0]), [1,1,1], true);
    this.labTex = this._tex3d(new Uint8Array([0]), [1,1,1], false);
    this.cmapTex = gl.createTexture(); this.lcolTex = gl.createTexture();
    this.setColormap("bone"); this.setLabelColors({});
    this.state = { yaw: 0.8, pitch: 0.45, dist: 1.7, mode: 0, thr: 0.35, dens: 1, wlo: 0, whi: 1,
      clipMin: [0,0,0], clipMax: [1,1,1], cutFaces: 1, hasVol: 0, hasLab: 0,
      box: [1,1,1], tex: [1,1,1] };
    this._mouse();
  }
  _program(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  _tex3d(data, [nx, ny, nz], linear, tex) {
    const gl = this.gl; tex = tex || gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    const f = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, f);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, f);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R])
      gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    return tex;
  }
  _tex2d(tex, rgba, n) {
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  }
  setColormap(name) {
    const stops = CMAPS[name] || CMAPS.bone, out = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const x = i / 255 * (stops.length - 1), k = Math.min(Math.floor(x), stops.length - 2), f = x - k;
      for (let c = 0; c < 3; c++) out[i*4+c] = stops[k][c] * (1-f) + stops[k+1][c] * f;
      out[i*4+3] = 255;
    }
    this._tex2d(this.cmapTex, out, 256); this.draw();
  }
  setLabelColors(map) { // {id: [r,g,b,a]}
    const out = new Uint8Array(256 * 4);
    for (const [id, c] of Object.entries(map)) out.set(c, (+id) * 4);
    this._tex2d(this.lcolTex, out, 256);
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.lcolTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.draw();
  }
  _box(shape, vox) {
    const ext = [shape[2]*vox[2], shape[1]*vox[1], shape[0]*vox[0]], m = Math.max(...ext);
    return ext.map(e => e / m);
  }
  setVolume(shape, vox, bytes) {
    this._tex3d(bytes, [shape[2], shape[1], shape[0]], true, this.volTex);
    Object.assign(this.state, { hasVol: 1, box: this._box(shape, vox), tex: [shape[2], shape[1], shape[0]] });
    this.draw();
  }
  setLabels(shape, vox, bytes) {
    if (!bytes) { this.state.hasLab = 0; this.draw(); return; }
    this._tex3d(bytes, [shape[2], shape[1], shape[0]], false, this.labTex);
    this.state.hasLab = 1;
    if (!this.state.hasVol) Object.assign(this.state, { box: this._box(shape, vox), tex: [shape[2], shape[1], shape[0]] });
    this.draw();
  }
  _mouse() {
    let drag = null;
    this.canvas.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; this.canvas.setPointerCapture(e.pointerId); });
    this.canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      this.state.yaw -= (e.clientX - drag[0]) * 0.008;
      this.state.pitch = Math.max(-1.5, Math.min(1.5, this.state.pitch + (e.clientY - drag[1]) * 0.008));
      drag = [e.clientX, e.clientY]; this.draw();
    });
    this.canvas.addEventListener("pointerup", () => drag = null);
    this.canvas.addEventListener("wheel", e => { e.preventDefault();
      this.state.dist = Math.max(0.6, Math.min(8, this.state.dist * Math.exp(e.deltaY * 0.001))); this.draw(); },
      { passive: false });
  }
  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this._draw(); });
  }
  _draw() {
    const gl = this.gl, c = this.canvas, s = this.state, dpr = Math.min(devicePixelRatio, 2);
    const w = Math.max(1, c.clientWidth * dpr | 0), h = Math.max(1, c.clientHeight * dpr | 0);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.02, 0.027, 0.047, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!s.hasVol && !s.hasLab) return;
    const eye = [s.dist * Math.cos(s.pitch) * Math.sin(s.yaw), s.dist * Math.sin(s.pitch),
                 s.dist * Math.cos(s.pitch) * Math.cos(s.yaw)];
    const mvp = mul(perspective(0.6, w / h, 0.05, 50), lookAt(eye, [0,0,0], [0,1,0]));
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniformMatrix4fv(this.u.mvp, false, mvp);
    gl.uniform3fv(this.u.boxSize, s.box); gl.uniform3fv(this.u.camPos, eye);
    gl.uniform3fv(this.u.clipMin, s.clipMin); gl.uniform3fv(this.u.clipMax, s.clipMax);
    gl.uniform3fv(this.u.texSize, s.tex);
    gl.uniform1i(this.u.mode, s.mode); gl.uniform1i(this.u.hasVol, s.hasVol);
    gl.uniform1i(this.u.hasLab, s.hasLab); gl.uniform1i(this.u.cutFaces, s.cutFaces);
    gl.uniform1f(this.u.thr, s.thr); gl.uniform1f(this.u.dens, s.dens);
    gl.uniform1f(this.u.wlo, s.wlo); gl.uniform1f(this.u.whi, s.whi);
    const bind = (unit, target, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(target, tex); gl.uniform1i(this.u[name], unit); };
    bind(0, gl.TEXTURE_3D, this.volTex, "vol"); bind(1, gl.TEXTURE_3D, this.labTex, "lab");
    bind(2, gl.TEXTURE_2D, this.cmapTex, "cmap"); bind(3, gl.TEXTURE_2D, this.lcolTex, "lcol");
    gl.drawArrays(gl.TRIANGLES, 0, 36);
  }
}

// ---- tiny mat4 helpers (column-major) ----
function perspective(fovy, aspect, n, f) {
  const t = 1 / Math.tan(fovy / 2), nf = 1 / (n - f);
  return new Float32Array([t/aspect,0,0,0, 0,t,0,0, 0,0,(f+n)*nf,-1, 0,0,2*f*n*nf,0]);
}
function lookAt(e, c, up) {
  const z = norm(sub(e, c)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -dot(x,e), -dot(y,e), -dot(z,e), 1]);
}
function mul(a, b) { const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0;
    for (let k = 0; k < 4; k++) s += a[k*4+j] * b[i*4+k]; o[i*4+j] = s; } return o; }
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = a => { const l = Math.hypot(...a) || 1; return a.map(v => v / l); };

window.VolumeRenderer = VolumeRenderer;
