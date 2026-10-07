import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';

const host = document.querySelector('#view');
const viewport = document.querySelector('#viewport');
const statusEl = document.querySelector('#status');
const itemList = document.querySelector('#items');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b1118');
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100000);
camera.up.set(0, 0, 1); // source mesh uses local ENU: east, north, up
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
host.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = 'absolute';
labels.domElement.style.inset = '0';
labels.domElement.style.pointerEvents = 'none';
labels.domElement.className = 'css2d-renderer';
host.appendChild(labels.domElement);
const gapFillCanvas = document.createElement('canvas');
gapFillCanvas.id = 'gapFillOverlay';
const gapFillContext = gapFillCanvas.getContext('2d', { alpha: true });
host.insertBefore(gapFillCanvas, labels.domElement);

scene.add(new THREE.HemisphereLight(0xdcecff, 0x26313a, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-1, -1, 3);
scene.add(sun);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.target.set(0, 0, 0);
controls.update();

let model = null;
let modelBox = null;
let mode = '';
let markerCounter = 1;
let annotationData = [];
let loadedName = '';
let currentOriginSource = 'bundled DJI0004 SRT origin';
let projectOriginMetadata = null;
let dragStart = null;
let cameraTrack = [];
let cameraMarkers = [];
let activeCameraIndex = 0;
let flightViewActive = false;
let orbitCameraState = null;
let autoGapFillEnabled = true;
let gapFillDirty = true;
let lastGapFillUpdate = 0;
let gapRenderTarget = null;
const gapMaskMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, toneMapped: false });
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const enuGrid = new THREE.GridHelper(100, 20, 0x3d7283, 0x243b48);
// GridHelper lies in XZ; rotate it into the ENU east/north plane.
enuGrid.rotation.x = Math.PI / 2;
enuGrid.material.transparent = true;
enuGrid.material.opacity = 0.35;
scene.add(enuGrid);
const axes = new THREE.AxesHelper(10);
scene.add(axes);

function setStatus(message) { statusEl.textContent = message; }
function origin() {
  const latText = document.querySelector('#lat0').value.trim();
  const lonText = document.querySelector('#lon0').value.trim();
  if (!latText || !lonText) return null;
  const lat = Number(latText);
  const lon = Number(lonText);
  return Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180 ? { lat, lon } : null;
}
function geoFromEnu(x, y) {
  const o = origin();
  if (!o) return null;
  const lat = o.lat + y / 110540.0;
  const metersPerLonDegree = 110540.0 * Math.cos(THREE.MathUtils.degToRad(o.lat));
  const lon = o.lon + x / metersPerLonDegree;
  return { lat, lon };
}
function enuFromGeo(lat, lon) {
  const o = origin();
  if (!o) return null;
  return {
    x: (lon - o.lon) * 110540.0 * Math.cos(THREE.MathUtils.degToRad(o.lat)),
    y: (lat - o.lat) * 110540.0,
  };
}
function validGps(lat, lon) {
  return Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180 && (lat !== 0 || lon !== 0);
}
function parseSrtTelemetry(text) {
  const blocks=text.replace(/^\uFEFF/,'').split(/\r?\n\s*\r?\n/);
  const samples=[];
  for(const block of blocks) {
    const time=block.match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->/);
    const lat=block.match(/\[\s*latitude\s*:\s*([-+\d.]+)/i);
    const lon=block.match(/\[\s*longitude\s*:\s*([-+\d.]+)/i);
    const alt=block.match(/\[\s*rel_alt\s*:\s*([-+\d.]+)/i) || block.match(/\[\s*altitude\s*:\s*([-+\d.]+)/i);
    if(!lat||!lon) continue;
    const latitude=Number(lat[1]),longitude=Number(lon[1]);
    if(!validGps(latitude,longitude)) continue;
    const t=time?((Number(time[1])*3600+Number(time[2])*60+Number(time[3])+Number(time[4])/1000)):samples.length;
    samples.push({lat:latitude,lon:longitude,alt:alt?Number(alt[1]):null,t});
  }
  return samples;
}
function splitCsvLine(line) {
  const values=[]; let value=''; let quoted=false;
  for(let i=0;i<line.length;i++) {
    const c=line[i];
    if(c==='"' && quoted && line[i+1]==='"'){value+='"';i++;}
    else if(c==='"') quoted=!quoted;
    else if(c===',' && !quoted){values.push(value.trim());value='';}
    else value+=c;
  }
  values.push(value.trim()); return values;
}
function parseCsvTelemetry(text) {
  const lines=text.replace(/^\uFEFF/,'').split(/\r?\n/);
  for(let h=0;h<Math.min(lines.length,100);h++) {
    const header=splitCsvLine(lines[h]).map(v=>v.replace(/^"|"$/g,'').trim().toLowerCase());
    const latI=header.findIndex(v=>v==='latitude'||v==='osd.latitude'||v.endsWith('.latitude'));
    const lonI=header.findIndex(v=>v==='longitude'||v==='osd.longitude'||v.endsWith('.longitude'));
    if(latI<0||lonI<0) continue;
    const altI=header.findIndex(v=>v.includes('rel_alt')||v.includes('height_above_takeoff')||v==='altitude'||v==='osd.height [m]'||v==='height_m');
    const timeI=header.findIndex(v=>v==='time'||v.includes('timestamp')||v.includes('datetime')||v.includes('updateTime'.toLowerCase()));
    const samples=[];
    for(let r=h+1;r<lines.length;r++) {
      if(!lines[r].trim()) continue;
      const cols=splitCsvLine(lines[r]);
      const lat=Number((cols[latI]||'').replace(/^"|"$/g,''));
      const lon=Number((cols[lonI]||'').replace(/^"|"$/g,''));
      const altText=altI>=0?(cols[altI]||'').replace(/^"|"$/g,'').trim():'';
      const alt=altText?Number(altText):null;
      if(validGps(lat,lon)) samples.push({lat,lon,alt:Number.isFinite(alt)?alt:null,t:timeI>=0?(cols[timeI]||''):samples.length});
    }
    return samples;
  }
  return [];
}
function downsampleTrack(samples, maxCount=24) {
  if(samples.length<=maxCount) return samples;
  const out=[];
  for(let i=0;i<maxCount;i++) out.push(samples[Math.round(i*(samples.length-1)/(maxCount-1))]);
  return out;
}
function logOrigin(samples, logFile) {
  const first=samples.find(p=>validGps(p.lat,p.lon));
  return first?{lat:first.lat,lon:first.lon,sample:`first valid ${/\.srt$/i.test(logFile.name)?'SRT GPS sample':'CSV GPS row'}`}:null;
}
function applyLogOrigin(logFile, parsed) {
  document.querySelector('#lat0').value=parsed.lat;
  document.querySelector('#lon0').value=parsed.lon;
  currentOriginSource=`${logFile.name} · ${parsed.sample}`;
  const status=document.querySelector('#originStatus');
  status.textContent=`Location ready · ${currentOriginSource} · ${parsed.lat.toFixed(6)}°N, ${parsed.lon.toFixed(6)}°E`;
  status.style.color='#9fe1ee';
  refreshAnnotationTags(); refreshList(); rebuildCameraMarkers();
}
async function handleAssetFiles(fileList) {
  const files=[...fileList];
  const modelFile=files.find(f=>/\.(glb|obj)$/i.test(f.name));
  const logFile=files.find(f=>/\.(srt|csv)$/i.test(f.name));
  if(logFile) {
    try {
      const text=await logFile.text();
      const rawSamples=/\.srt$/i.test(logFile.name)?parseSrtTelemetry(text):parseCsvTelemetry(text);
      const parsed=logOrigin(rawSamples,logFile);
      setCameraTrack(rawSamples,`${logFile.name} GPS track`);
      if(parsed) applyLogOrigin(logFile,parsed);
      else {
        if(projectOriginMetadata) {
          document.querySelector('#lat0').value=projectOriginMetadata.origin_lat;
          document.querySelector('#lon0').value=projectOriginMetadata.origin_lon;
          currentOriginSource=`${projectOriginMetadata.source_file} · ${projectOriginMetadata.source_sample}`;
        }
        const originStatus=document.querySelector('#originStatus');
        originStatus.textContent=`Could not read ${logFile.name}; using bundled DJI0004 location.`;
        originStatus.style.color='#ffbf66';
        setStatus(`Could not read GPS from ${logFile.name}; using the bundled DJI0004 fallback if available.`);
      }
    } catch(err) { setStatus(`Could not read ${logFile.name}: ${err.message}`); }
  } else if(projectOriginMetadata) {
    document.querySelector('#lat0').value=projectOriginMetadata.origin_lat;
    document.querySelector('#lon0').value=projectOriginMetadata.origin_lon;
    currentOriginSource=`${projectOriginMetadata.source_file} · ${projectOriginMetadata.source_sample}`;
    document.querySelector('#originStatus').textContent=`Location ready · bundled ${currentOriginSource} · ${projectOriginMetadata.origin_lat.toFixed(6)}°N, ${projectOriginMetadata.origin_lon.toFixed(6)}°E`;
    document.querySelector('#originStatus').style.color='#9fe1ee';
  }
  if(modelFile) await loadModelFile(modelFile);
  else setStatus(logFile?'Location reference loaded. Now choose the model file.':'Choose a .glb/.obj model. Matching .srt/.csv is optional.');
}
async function loadProjectOrigin() {
  const status = document.querySelector('#originStatus');
  try {
    const response = await fetch('./geotwin_origin.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`origin metadata returned HTTP ${response.status}`);
    const metadata = await response.json();
    projectOriginMetadata = metadata;
    currentOriginSource = `${metadata.source_file || 'project telemetry'} · ${metadata.source_sample || 'matching DJI telemetry origin'}`;
    if (!Number.isFinite(metadata.origin_lat) || !Number.isFinite(metadata.origin_lon)) throw new Error('origin metadata has no valid latitude/longitude');
    document.querySelector('#lat0').value = metadata.origin_lat;
    document.querySelector('#lon0').value = metadata.origin_lon;
    const source = metadata.source_file || 'project telemetry';
    const tag = metadata.source_sample || 'matching DJI telemetry origin';
    status.textContent = `Location ready · ${source} · ${tag} · ${metadata.origin_lat.toFixed(6)}°N, ${metadata.origin_lon.toFixed(6)}°E`;
    status.style.color = '#9fe1ee';
    document.querySelector('#frameBadge').textContent = 'Approximate WGS84 location ready';
    refreshAnnotationTags(); refreshList();
    try {
      const trackResponse=await fetch('./geotwin_camera_path.json',{cache:'no-store'});
      if(trackResponse.ok) {
        const trackData=await trackResponse.json();
        if(Array.isArray(trackData.samples)) setCameraTrack(trackData.samples,trackData.source_file||'bundled DJI0004 SRT');
      }
    } catch(err) { console.warn('Bundled camera track unavailable:',err); }
  } catch (err) {
    console.error(err);
    status.textContent = 'No default location found. Choose the matching DJI SRT/CSV, or use Advanced to enter its origin.';
    status.style.color = '#ffbf66';
  }
}
function makeTag(text, inferred = false) {
  const element = document.createElement('div');
  element.className = `tag${inferred ? ' inferred' : ''}`;
  element.textContent = text;
  return new CSS2DObject(element);
}
function formatCoord(point) {
  const geo = geoFromEnu(point.x, point.y);
  if (!geo) return 'location not loaded';
  return `${geo.lat.toFixed(5)}°N ${geo.lon.toFixed(5)}°E`;
}
function markerRadius() {
  return modelBox ? Math.max(modelBox.getSize(new THREE.Vector3()).length() * 0.003, 0.45) : 0.7;
}
function addLandmark(item) {
  const p = new THREE.Vector3(...item.xyz);
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(markerRadius(), 16, 12),
    new THREE.MeshStandardMaterial({ color: 0x54c8e8, emissive: 0x105065, roughness: 0.35 })
  );
  sphere.position.copy(p);
  sphere.userData.annotationId = item.id;
  scene.add(sphere);
  const tag = makeTag(`${item.name} · ${formatCoord(p)}`);
  tag.position.copy(p).add(new THREE.Vector3(0, 0, markerRadius() * 1.7));
  tag.userData.annotationId = item.id;
  scene.add(tag);
  item._objects = [sphere, tag];
}
function addPatch(item) {
  const half = item.width / 2;
  const z = item.xyz[2] + 0.04;
  const vertices = new Float32Array([
    item.xyz[0]-half,item.xyz[1]-half,z, item.xyz[0]+half,item.xyz[1]-half,z,
    item.xyz[0]+half,item.xyz[1]+half,z, item.xyz[0]-half,item.xyz[1]-half,z,
    item.xyz[0]+half,item.xyz[1]+half,z, item.xyz[0]-half,item.xyz[1]+half,z,
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  const fill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0xffb547, transparent: true, opacity: 0.32, side: THREE.DoubleSide,
    depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
  }));
  fill.userData.annotationId = item.id;
  scene.add(fill);
  const borderGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(item.xyz[0]-half,item.xyz[1]-half,z),
    new THREE.Vector3(item.xyz[0]+half,item.xyz[1]-half,z),
    new THREE.Vector3(item.xyz[0]+half,item.xyz[1]+half,z),
    new THREE.Vector3(item.xyz[0]-half,item.xyz[1]+half,z),
    new THREE.Vector3(item.xyz[0]-half,item.xyz[1]-half,z),
  ]);
  const border = new THREE.Line(borderGeom, new THREE.LineBasicMaterial({ color: 0xffb547 }));
  border.userData.annotationId = item.id;
  scene.add(border);
  const tag = makeTag(`${item.name} · ${formatCoord(new THREE.Vector3(...item.xyz))}`, true);
  tag.position.set(item.xyz[0], item.xyz[1], z + Math.max(item.width * 0.12, 0.5));
  tag.userData.annotationId = item.id;
  scene.add(tag);
  item._objects = [fill, border, tag];
}
function addAnnotation(item) {
  annotationData.push(item);
  item.type === 'patch' ? addPatch(item) : addLandmark(item);
  refreshList();
}
function refreshAnnotationTags() {
  for (const item of annotationData) {
    if (!item._objects) continue;
    const tag = item._objects[item._objects.length - 1];
    tag.element.textContent = `${item.name} · ${formatCoord(new THREE.Vector3(...item.xyz))}`;
  }
}
function refreshList() {
  itemList.replaceChildren();
  if (!annotationData.length) { itemList.textContent = 'No annotations yet.'; return; }
  for (const item of annotationData) {
    const row = document.createElement('div'); row.className = 'item';
    const details = document.createElement('span');
    details.textContent = `${item.type === 'patch' ? 'INFERRED' : item.kind} · ${item.name} · ${formatCoord(new THREE.Vector3(...item.xyz))}`;
    const remove = document.createElement('button'); remove.textContent = '×'; remove.title = 'Remove';
    remove.addEventListener('click', () => removeAnnotation(item.id));
    row.append(details, remove); itemList.append(row);
  }
}
function removeAnnotation(id) {
  const idx = annotationData.findIndex(a => a.id === id);
  if (idx < 0) return;
  for (const object of annotationData[idx]._objects || []) {
    scene.remove(object);
    object.traverse?.(child => { child.geometry?.dispose?.(); child.material?.dispose?.(); });
  }
  annotationData.splice(idx, 1);
  refreshList();
}
function getMeshHits(clientX, clientY) {
  if (!model) return [];
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  return raycaster.intersectObject(model, true);
}
function clearGapFill() {
  gapFillContext.clearRect(0, 0, gapFillCanvas.width, gapFillCanvas.height);
}
function convexHullForLargestSurface(surface, w, h) {
  const total = w * h, visited = new Uint8Array(total), queue = new Int32Array(total);
  let largestCount = 0;
  const components = [];
  for (let start = 0; start < total; start++) {
    if (!surface[start] || visited[start]) continue;
    let head = 0, tail = 0; queue[tail++] = start; visited[start] = 1;
    while (head < tail) {
      const i = queue[head++], x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const nx=x+dx, ny=y+dy;
        if(nx<0||nx>=w||ny<0||ny>=h)continue;
        const j=ny*w+nx;
        if(surface[j]&&!visited[j]){visited[j]=1;queue[tail++]=j;}
      }
    }
    largestCount = Math.max(largestCount, tail);
    components.push(queue.slice(0, tail));
  }
  if (!largestCount || largestCount < 12) return null;

  const boundary=[];
  // The scan often projects one physical neighborhood into several screen
  // components because of occlusion and imperfect mesh connectivity. Build
  // the footprint from all substantial pieces so open gaps between buildings
  // and roads are included, while isolated raster speckles are ignored.
  const minComponent = Math.max(12, Math.floor(largestCount * 0.0005));
  for(const component of components){
    if(component.length < minComponent) continue;
    for(const i of component){
      const x=i%w,y=(i/w)|0;
      if(x===0||x===w-1||y===0||y===h-1||!surface[i-1]||!surface[i+1]||!surface[i-w]||!surface[i+w])
        boundary.push({x,y});
    }
  }
  const stride=Math.max(1,Math.ceil(boundary.length/3000));
  const points=boundary.filter((_,i)=>i%stride===0).sort((a,b)=>a.x-b.x||a.y-b.y);
  if(points.length<3)return null;
  const cross=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
  const lower=[];
  for(const q of points){while(lower.length>=2&&cross(lower[lower.length-2],lower[lower.length-1],q)<=0)lower.pop();lower.push(q);}
  const upper=[];
  for(let i=points.length-1;i>=0;i--){const q=points[i];while(upper.length>=2&&cross(upper[upper.length-2],upper[upper.length-1],q)<=0)upper.pop();upper.push(q);}
  lower.pop();upper.pop();const hull=lower.concat(upper);
  if(hull.length<3)return null;

  // Rasterize the dominant projected surface's convex hull. This covers gaps
  // that are open to the screen background as well as enclosed holes.
  const inside=new Uint8Array(total);let area=0;
  for(let y=0;y<h;y++){
    const scanY=y+0.5, xs=[];
    for(let i=0;i<hull.length;i++){
      const a=hull[i],b=hull[(i+1)%hull.length];
      if((a.y<=scanY&&b.y>scanY)||(b.y<=scanY&&a.y>scanY))
        xs.push(a.x+(scanY-a.y)*(b.x-a.x)/(b.y-a.y));
    }
    xs.sort((a,b)=>a-b);
    for(let k=0;k+1<xs.length;k+=2){
      const left=Math.max(0,Math.ceil(xs[k])),right=Math.min(w-1,Math.floor(xs[k+1]));
      for(let x=left;x<=right;x++){const j=y*w+x;if(!inside[j]){inside[j]=1;area++;}}
    }
  }
  return {mask:inside,area};
}
function updateGapFill() {
  if (!autoGapFillEnabled || !model) return;
  const screenW = renderer.domElement.width, screenH = renderer.domElement.height;
  if (!screenW || !screenH) return;
  const w = Math.min(960, screenW), h = Math.max(1, Math.round(w * screenH / screenW));
  if (!gapRenderTarget || gapRenderTarget.width !== w || gapRenderTarget.height !== h) {
    gapRenderTarget?.dispose();
    gapRenderTarget = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true, stencilBuffer: false });
  }
  const oldBackground = scene.background, oldOverride = scene.overrideMaterial, oldToneMapping = renderer.toneMapping;
  const oldRenderTarget = renderer.getRenderTarget();
  const rootVisibility = scene.children.map(object => [object, object.visible]);
  try {
    // Capture only model colors so the adjacent-color mean is not contaminated
    // by the grid, camera markers, or annotation labels.
    for (const object of scene.children) object.visible = object === model;
    renderer.setRenderTarget(gapRenderTarget);
    renderer.render(scene, camera);
    const colorPixels = new Uint8Array(w * h * 4);
    renderer.readRenderTargetPixels(gapRenderTarget, 0, 0, w, h, colorPixels);
    scene.background = new THREE.Color(0x000000);
    scene.overrideMaterial = gapMaskMaterial;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setRenderTarget(gapRenderTarget);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    const maskPixels = new Uint8Array(w * h * 4);
    renderer.readRenderTargetPixels(gapRenderTarget, 0, 0, w, h, maskPixels);
    const isSurface = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const src=(h-1-y)*w*4,row=y*w;
      for(let x=0;x<w;x++)isSurface[row+x]=maskPixels[src+x*4]>96?1:0;
    }
    const hull=convexHullForLargestSurface(isSurface,w,h);
    if(!hull||hull.area<32){
      clearGapFill();document.querySelector('#autoGapFill').textContent='Hide inferred gap fill';
      document.querySelector('#inferredBadge').textContent='INFERRED VIEW FILL · no sizeable scene footprint';
      gapFillDirty=false;lastGapFillUpdate=performance.now();return;
    }

    // Any non-surface pixel inside the main projected footprint is a visible
    // coverage gap. This includes open-to-background holes the previous
    // enclosed-component detector discarded.
    const gap=new Uint8Array(w*h);
    for(let i=0;i<gap.length;i++)gap[i]=hull.mask[i]&&!isSurface[i]?1:0;
    const visited=new Uint8Array(w*h),queue=new Int32Array(w*h),accepted=[];
    let gapPixels=0;
    for(let start=0;start<gap.length;start++){
      if(!gap[start]||visited[start])continue;
      let head=0,tail=0;queue[tail++]=start;visited[start]=1;
      while(head<tail){
        const i=queue[head++],x=i%w,y=(i/w)|0;
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
          if(!dx&&!dy)continue;const nx=x+dx,ny=y+dy;if(nx<0||nx>=w||ny<0||ny>=h)continue;
          const j=ny*w+nx;if(gap[j]&&!visited[j]){visited[j]=1;queue[tail++]=j;}
        }
      }
      // Keep separate large open areas bounded: a patch may cover up to 70%
      // of the dominant model footprint, but never the entire background.
      if(tail>=8&&tail<=hull.area*0.70){accepted.push(queue.slice(0,tail));gapPixels+=tail;}
    }
    const image=gapFillContext.createImageData(w,h);
    const colorAt=(i,c)=>colorPixels[((h-1-((i/w)|0))*w+(i%w))*4+c];
    for(const pixels of accepted){
      const edge=new Set();
      for(const i of pixels){
        const x=i%w,y=(i/w)|0;
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
          if(!dx&&!dy)continue;const nx=x+dx,ny=y+dy;if(nx<0||nx>=w||ny<0||ny>=h)continue;
          const j=ny*w+nx;if(isSurface[j])edge.add(j);
        }
      }
      if(!edge.size)continue;
      let r=0,g=0,b=0;
      for(const i of edge){r+=colorAt(i,0);g+=colorAt(i,1);b+=colorAt(i,2);}
      r=Math.round(r/edge.size);g=Math.round(g/edge.size);b=Math.round(b/edge.size);
      for(const i of pixels){
        const o=i*4,x=i%w,y=(i/w)|0;
        if((x+y)%12<2){image.data[o]=Math.round(r*0.55+255*0.45);image.data[o+1]=Math.round(g*0.55+181*0.45);image.data[o+2]=Math.round(b*0.55+71*0.45);}
        else{image.data[o]=r;image.data[o+1]=g;image.data[o+2]=b;}
        image.data[o+3]=232;
      }
    }
    gapFillCanvas.width=w;gapFillCanvas.height=h;gapFillContext.putImageData(image,0,0);
    document.querySelector('#autoGapFill').textContent=`Hide inferred gap fill (${accepted.length} areas)`;
    document.querySelector('#inferredBadge').textContent=`INFERRED VIEW FILL · ${gapPixels.toLocaleString()} px · NOT CAMERA OBSERVED`;
    gapFillDirty=false;lastGapFillUpdate=performance.now();
  }catch(error){
    console.warn('Could not update view-dependent gap fill:',error);clearGapFill();
    gapFillDirty=false;lastGapFillUpdate=performance.now();
    setStatus(`Automatic gap fill could not render: ${error.message||error}`);
  }finally{
    renderer.setRenderTarget(oldRenderTarget);
    for(const [object,visible] of rootVisibility)object.visible=visible;
    scene.background=oldBackground;scene.overrideMaterial=oldOverride;renderer.toneMapping=oldToneMapping;
  }
}
function clickPosition(clientX, clientY, useFallback = false) {
  const hits = getMeshHits(clientX, clientY);
  if (hits.length) return hits[0].point.clone();
  if (!useFallback) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const z = Number(document.querySelector('#fallbackZ').value);
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -z);
  const point = new THREE.Vector3();
  return raycaster.ray.intersectPlane(plane, point) ? point : null;
}
function placeAt(clientX, clientY) {
  if (!origin()) { setStatus('Choose a matching SRT/CSV or check the bundled location reference.'); return; }
  if (!model) { setStatus('Load the reviewed GLB/OBJ first.'); return; }
  if (mode === 'landmark') {
    const p = clickPosition(clientX, clientY, false);
    if (!p) { setStatus('Click on the mesh surface to place this landmark.'); return; }
    const label = document.querySelector('#labelText').value.trim() || `Feature ${markerCounter}`;
    const kind = document.querySelector('#kind').value;
    addAnnotation({ id: `landmark-${Date.now()}-${markerCounter}`, type: 'landmark', name: label, kind,
      xyz: p.toArray(), status: 'observed_surface_annotation' });
    markerCounter++;
    document.querySelector('#labelText').value = `Building ${String(markerCounter).padStart(2, '0')}`;
    setStatus(`Landmark placed at ${formatCoord(p)}. Coordinates are approximate.`);
  } else if (mode === 'setz') {
    const p = clickPosition(clientX, clientY, false);
    if (!p) { setStatus('Click a visible road/ground surface to set the inferred-patch fallback height.'); return; }
    document.querySelector('#fallbackZ').value = p.z.toFixed(2);
    setStatus(`Fallback patch plane set to Z=${p.z.toFixed(2)} m from the clicked observed surface.`);
  } else if (mode === 'patch') {
    const p = clickPosition(clientX, clientY, true);
    if (!p) { setStatus('Could not place patch here; adjust view or fallback Z.'); return; }
    const label = document.querySelector('#patchName').value.trim() || 'Inferred fill — not camera observed';
    const width = THREE.MathUtils.clamp(Number(document.querySelector('#patchSize').value) || 8, 1, 100);
    addAnnotation({ id: `patch-${Date.now()}-${markerCounter}`, type: 'patch', name: label, kind: 'coverage_gap',
      xyz: p.toArray(), width, status: 'inferred_not_camera_observed' });
    markerCounter++;
    setStatus(`Amber inferred patch added at ${formatCoord(p)}. It is separate from observed geometry.`);
  }
}
function setMode(next) {
  mode = mode === next ? '' : next;
  document.querySelector('#landmarkMode').classList.toggle('active', mode === 'landmark');
  document.querySelector('#patchMode').classList.toggle('active', mode === 'patch');
  document.querySelector('#setFallbackMode').classList.toggle('active', mode === 'setz');
  viewport.style.cursor = mode ? 'crosshair' : 'grab';
  setStatus(mode === 'landmark' ? 'Click a visible mesh surface to pin a landmark.' :
    mode === 'patch' ? 'Click a gap; if no mesh is under the cursor, patch uses fallback Z.' :
    mode === 'setz' ? 'Click an observed road/ground area to copy its surface height.' : 'Orbit, pan, and zoom the model.');
}
function focusModel(object) {
  modelBox = new THREE.Box3().setFromObject(object);
  const center = modelBox.getCenter(new THREE.Vector3());
  const size = modelBox.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 1);
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.35, center.y - radius * 1.55, center.z + radius * 0.95);
  camera.near = Math.max(radius / 10000, 0.01); camera.far = radius * 100;
  camera.updateProjectionMatrix(); controls.update();
  enuGrid.position.set(center.x, center.y, modelBox.min.z - Math.max(size.z * 0.02, 1));
  enuGrid.scale.setScalar(Math.max(size.x, size.y, 20) / 100);
  axes.position.set(center.x, center.y, modelBox.min.z);
  document.querySelector('#fallbackZ').value = modelBox.min.z.toFixed(2);
  rebuildCameraMarkers();
}
let cameraTrackSource='bundled DJI0004 telemetry';
function removeCameraMarkers() {
  for(const object of cameraMarkers) {
    scene.remove(object);
    object.traverse?.(child=>{child.geometry?.dispose?.();child.material?.dispose?.();});
  }
  cameraMarkers=[];
}
function cameraWorldPoints() {
  const o=origin();
  if(!o||!modelBox||!cameraTrack.length)return [];
  const size=modelBox.getSize(new THREE.Vector3());
  const altitudes=cameraTrack.map(p=>p.alt).filter(Number.isFinite).sort((a,b)=>a-b);
  const midAlt=altitudes.length?altitudes[Math.floor(altitudes.length/2)]:null;
  const clearance=midAlt===null?Math.max(30,size.z*0.35):THREE.MathUtils.clamp(midAlt,15,120);
  const baseZ=modelBox.max.z+clearance;
  return cameraTrack.map(sample=>{
    const xy=enuFromGeo(sample.lat,sample.lon);
    const dz=Number.isFinite(sample.alt)&&midAlt!==null?sample.alt-midAlt:0;
    return new THREE.Vector3(xy.x,xy.y,baseZ+dz);
  });
}
function updateCameraButtons() {
  const usable=flightViewActive&&cameraTrack.length>1;
  document.querySelector('#previousCamera').disabled=!usable;
  document.querySelector('#nextCamera').disabled=!usable;
}
function moveToCameraView(index) {
  if(!model||!cameraTrack.length||!modelBox)return;
  const points=cameraWorldPoints();
  activeCameraIndex=(index+points.length)%points.length;
  const target=modelBox.getCenter(new THREE.Vector3());
  const position=points[activeCameraIndex];
  camera.position.copy(position);camera.lookAt(target);controls.target.copy(target);
  const distance=position.distanceTo(target);
  camera.near=Math.max(distance/10000,0.01);camera.far=Math.max(distance*100,1000);
  camera.updateProjectionMatrix();controls.update();
  const group=cameraMarkers[0];
  if(group)group.children.forEach(child=>{if(child.userData.cameraIndex!==undefined)child.material.color.set(child.userData.cameraIndex===activeCameraIndex?0xffb547:0x54c8e8);});
  const sample=cameraTrack[activeCameraIndex];
  document.querySelector('#cameraViewStatus').textContent=`Drone location ${activeCameraIndex+1} of ${cameraTrack.length} · ${sample.lat.toFixed(6)}°N, ${sample.lon.toFixed(6)}°E · aimed at the model center (approximate).`;
}
function rebuildCameraMarkers() {
  removeCameraMarkers();
  const points=cameraWorldPoints();
  const toggle=document.querySelector('#flightViewToggle');
  const status=document.querySelector('#cameraViewStatus');
  if(points.length<2) {
    toggle.disabled=true;updateCameraButtons();
    if(!cameraTrack.length)status.textContent='Load the matching SRT/CSV to use the recorded drone locations.';
    return;
  }
  const group=new THREE.Group();group.name='Approximate DJI flight path';group.visible=flightViewActive;scene.add(group);cameraMarkers.push(group);
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0x54c8e8,transparent:true,opacity:0.72})));
  const radius=markerRadius()*1.8;
  points.forEach((p,i)=>{
    const marker=new THREE.Mesh(new THREE.SphereGeometry(radius,10,8),new THREE.MeshBasicMaterial({color:i===activeCameraIndex?0xffb547:0x54c8e8}));
    marker.position.copy(p);marker.userData.cameraIndex=i;group.add(marker);
  });
  toggle.disabled=false;status.textContent=`${points.length} sampled drone positions · ${cameraTrackSource}. Height and viewing direction are approximate.`;
  updateCameraButtons();
  if(flightViewActive)moveToCameraView(activeCameraIndex);
}
function setCameraTrack(samples,source) {
  cameraTrack=downsampleTrack(samples.filter(p=>validGps(p.lat,p.lon)),24);
  cameraTrackSource=source;activeCameraIndex=0;rebuildCameraMarkers();
}
function toggleFlightView() {
  if(!model||cameraTrack.length<2)return;
  flightViewActive=!flightViewActive;
  if(flightViewActive) {
    orbitCameraState={position:camera.position.clone(),target:controls.target.clone()};
    document.querySelector('#flightViewToggle').textContent='Return to orbit view';
    for(const object of cameraMarkers)object.visible=true;
    moveToCameraView(activeCameraIndex);
  } else {
    document.querySelector('#flightViewToggle').textContent='Show drone camera views';
    for(const object of cameraMarkers)object.visible=false;
    if(orbitCameraState){camera.position.copy(orbitCameraState.position);controls.target.copy(orbitCameraState.target);camera.lookAt(controls.target);controls.update();orbitCameraState=null;}
    document.querySelector('#cameraViewStatus').textContent=`Flight route hidden · ${cameraTrack.length} sampled positions loaded.`;
  }
  updateCameraButtons();
}
function modelBoundsMatch(metadata) {
  if (!metadata?.expected_mesh_bounds || !modelBox) return null;
  const actualMin = modelBox.min.toArray(), actualMax = modelBox.max.toArray();
  const expectedMin = metadata.expected_mesh_bounds.min, expectedMax = metadata.expected_mesh_bounds.max;
  const error = Math.max(...actualMin.map((v,i)=>Math.abs(v-expectedMin[i])), ...actualMax.map((v,i)=>Math.abs(v-expectedMax[i])));
  return { match:error <= (metadata.bounds_tolerance_m ?? 0.5), error };
}
async function loadModelFile(file) {
  const isObj=/\.obj$/i.test(file.name);
  setStatus(`Loading ${file.name}…${isObj?' OBJ is large and may take a few minutes.':' large meshes may take a little while.'}`);
  document.querySelector('#modelStatus').textContent=`Loading ${file.name}…`;
  try {
    let loadedObject;
    if(isObj) {
      const text=await file.text();
      loadedObject=new OBJLoader().parse(text);
      loadedObject.traverse(obj=>{
        if(!obj.isMesh)return;
        const mats=Array.isArray(obj.material)?obj.material:[obj.material];
        for(const mat of mats) {
          mat.side=THREE.DoubleSide;
          if(obj.geometry?.getAttribute('color')) mat.vertexColors=true;
          mat.needsUpdate=true;
        }
      });
    } else {
      const buffer=await file.arrayBuffer();
      const gltf=await new GLTFLoader().parseAsync(buffer,'');
      loadedObject=gltf.scene;
      loadedObject.traverse(obj=>{
        if(!obj.isMesh)return;
        obj.frustumCulled=false;
        const mats=Array.isArray(obj.material)?obj.material:[obj.material];
        for(const mat of mats) {
          mat.side=THREE.DoubleSide;
          if(obj.geometry?.getAttribute('color')) mat.vertexColors=true;
          if(mat.map)mat.map.colorSpace=THREE.SRGBColorSpace;
          mat.needsUpdate=true;
        }
      });
    }
    if(model)scene.remove(model);
    model=loadedObject; scene.add(model); loadedName=file.name; focusModel(model);
    const fillButton=document.querySelector('#autoGapFill');
    fillButton.disabled=false;fillButton.classList.add('active');fillButton.textContent='Hide inferred gap fill';
    autoGapFillEnabled=true;gapFillCanvas.style.display='block';gapFillDirty=true;
    try {
      const meta=await (await fetch('./geotwin_origin.json',{cache:'no-store'})).json();
      const check=modelBoundsMatch(meta);
      if(check&&!check.match) {
        const warning=`Mesh differs from the bundled clean500 reference by ${check.error.toFixed(2)} m; check the matching SRT/CSV origin.`;
        document.querySelector('#frameBadge').textContent='Mesh reference differs';
        document.querySelector('#frameBadge').style.color='#ffbf66';
        document.querySelector('#modelStatus').textContent=warning;
        setStatus(`Loaded ${file.name}. ${warning}`);
      } else {
        document.querySelector('#frameBadge').textContent='Clean500 mesh · approximate WGS84 ready';
        document.querySelector('#frameBadge').style.color='#9fe1ee';
        document.querySelector('#modelStatus').textContent=`${file.name} loaded · clean500 geometry recognized · location from ${currentOriginSource}`;
        setStatus('Model and location ready. Add selected landmarks or inferred gap patches.');
      }
    } catch {
      document.querySelector('#modelStatus').textContent=`${file.name} loaded · geographic origin ${origin()?'ready':'needs a matching SRT/CSV'}`;
      setStatus(`Loaded ${file.name}. Add a marker tool to annotate.`);
    }
  } catch(err) {
    console.error(err);
    document.querySelector('#modelStatus').textContent=`Could not load ${file.name}: ${err.message||err}`;
    setStatus(`Could not load model: ${err.message||err}`);
  }
}function makeFeature(item) {
  const p = new THREE.Vector3(...item.xyz);
  const geo = geoFromEnu(p.x, p.y);
  if (item.type === 'patch') {
    const d = item.width / 2;
    const ring = [[p.x-d,p.y-d],[p.x+d,p.y-d],[p.x+d,p.y+d],[p.x-d,p.y+d],[p.x-d,p.y-d]];
    const coordinates = ring.map(([x,y]) => { const ll=geoFromEnu(x,y); return [ll.lon,ll.lat]; });
    return { type:'Feature', geometry:{ type:'Polygon', coordinates:[coordinates] }, properties:{
      id:item.id, name:item.name, category:item.kind, status:item.status, width_m:item.width, model_z_m:p.z,
      note:'Planar display placeholder only; not measured or camera-observed geometry.' } };
  }
  return { type:'Feature', geometry:{ type:'Point', coordinates:[geo.lon,geo.lat] }, properties:{
    id:item.id, name:item.name, category:item.kind, status:item.status, model_z_m:p.z,
    coordinate_quality:'approximate; metre-scale camera-to-telemetry alignment residual' } };
}
function sessionObject() {
  return { schema:'sih-geotwin-annotation-session-v1', model_file:loadedName, frame:'local ENU metres; X=east, Y=north, Z=up',
    origin:{ latitude:Number(document.querySelector('#lat0').value), longitude:Number(document.querySelector('#lon0').value),
      source:currentOriginSource, accuracy:'approximate, not survey-grade' },
    annotations:annotationData.map(({_objects,...item})=>item) };
}
function saveJson(name, value) {
  const blob = new Blob([JSON.stringify(value,null,2)], {type:'application/json'});
  const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1500);
}

document.querySelector('#assetFiles').addEventListener('change', e => { if(e.target.files.length) handleAssetFiles(e.target.files); });
document.querySelector('#applyOrigin').addEventListener('click', () => {
  const o=origin(); if(!o) { setStatus('Latitude must be −90…90 and longitude −180…180.'); return; }
  currentOriginSource='manual origin override';
  refreshAnnotationTags(); refreshList(); rebuildCameraMarkers();
  document.querySelector('#originStatus').textContent=`Custom location reference applied · ${o.lat.toFixed(6)}°N, ${o.lon.toFixed(6)}°E`;
  setStatus(`Origin set. Pins now show approximate WGS84 coordinates.`);
});
document.querySelector('#landmarkMode').addEventListener('click',()=>setMode('landmark'));
document.querySelector('#patchMode').addEventListener('click',()=>setMode('patch'));
document.querySelector('#setFallbackMode').addEventListener('click',()=>setMode('setz'));
document.querySelector('#autoGapFill').addEventListener('click',()=>{
  if(!model){setStatus('Load a model before filling visible gaps.');return;}
  autoGapFillEnabled=!autoGapFillEnabled;
  const button=document.querySelector('#autoGapFill');
  button.classList.toggle('active',autoGapFillEnabled);
  gapFillCanvas.style.display=autoGapFillEnabled?'block':'none';
  if(autoGapFillEnabled){gapFillDirty=true;setStatus('Blending adjacent surface colors into gaps in this view. The fill is visibly marked as inferred.');}
  else{clearGapFill();button.textContent='Show inferred gap fill';document.querySelector('#inferredBadge').textContent='INFERRED VIEW FILL · HIDDEN';setStatus('View-only inferred fill hidden. The observed mesh is unchanged.');}
});
document.querySelector('#flightViewToggle').addEventListener('click',toggleFlightView);
document.querySelector('#previousCamera').addEventListener('click',()=>moveToCameraView((activeCameraIndex-1+cameraTrack.length)%cameraTrack.length));
document.querySelector('#nextCamera').addEventListener('click',()=>moveToCameraView((activeCameraIndex+1)%cameraTrack.length));
document.querySelector('#exportGeo').addEventListener('click',()=>{
  if(!origin()) { setStatus('Set the matching origin before exporting GeoJSON.'); return; }
  const geojson={type:'FeatureCollection',name:'GeoTwin observed and inferred annotations',
    metadata:{coordinate_reference:'WGS84 longitude/latitude (EPSG:4326)', model_frame:'local ENU metres',
      origin:origin(), origin_source:currentOriginSource, vertical_values:'stored as model_z_m properties, not absolute elevation',
      accuracy:'Approximate georeferencing; not survey-grade. Inferred patches are not observed geometry.'},
    features:annotationData.map(makeFeature)};
  saveJson('geotwin_annotations.geojson',geojson);
});
document.querySelector('#exportSession').addEventListener('click',()=>{
  if(!origin()) { setStatus('Set the matching origin before saving.'); return; }
  saveJson('geotwin_session.json',sessionObject());
});
document.querySelector('#importSession').addEventListener('change',async e=>{
  const file=e.target.files[0]; if(!file)return;
  try {
    const data=JSON.parse(await file.text());
    if(data.schema!=='sih-geotwin-annotation-session-v1'||!Array.isArray(data.annotations)) throw new Error('Unsupported annotation session format.');
    if(data.origin){document.querySelector('#lat0').value=data.origin.latitude;document.querySelector('#lon0').value=data.origin.longitude;currentOriginSource=data.origin.source||'saved session origin';}
    for(const old of [...annotationData]) removeAnnotation(old.id);
    annotationData=[];
    for(const item of data.annotations) addAnnotation(item);
    refreshAnnotationTags(); setStatus(`Loaded ${annotationData.length} annotations. Load the same model to see them in 3D.`);
  } catch(err) { setStatus(`Could not load session: ${err.message}`); }
});

renderer.domElement.addEventListener('pointerdown', e=>{dragStart={x:e.clientX,y:e.clientY};});
renderer.domElement.addEventListener('pointerup', e=>{
  if(!dragStart)return; const d=Math.hypot(e.clientX-dragStart.x,e.clientY-dragStart.y); dragStart=null;
  if(d<4&&mode) placeAt(e.clientX,e.clientY);
});
for(const event of ['dragenter','dragover']) viewport.addEventListener(event,e=>{e.preventDefault();document.querySelector('#drop').style.display='grid';});
for(const event of ['dragleave','drop']) viewport.addEventListener(event,e=>{e.preventDefault();document.querySelector('#drop').style.display='none';});
viewport.addEventListener('drop',e=>{if(e.dataTransfer.files.length)handleAssetFiles(e.dataTransfer.files);});
controls.addEventListener('change',()=>{gapFillDirty=true;});
window.addEventListener('resize',resize);
function resize(){const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);labels.setSize(w,h);gapFillDirty=true;}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);labels.render(scene,camera);if(autoGapFillEnabled&&gapFillDirty&&performance.now()-lastGapFillUpdate>180)updateGapFill();}
async function loadAssetsFromQuery(){
  const params=new URLSearchParams(location.search),modelPath=params.get('model'),logPath=params.get('log');
  if(!modelPath)return;
  try{
    setStatus('Loading the run output model and flight log directly from Colab…');
    const fetchFile=async(path)=>{
      const url=new URL(path,location.href);
      if(url.origin!==location.origin)throw new Error('Viewer auto-load accepts files from this Colab run only.');
      const response=await fetch(url.toString());
      if(!response.ok)throw new Error(`Could not fetch ${url.pathname} (HTTP ${response.status}).`);
      const blob=await response.blob();
      const filename=decodeURIComponent(url.pathname.split('/').pop()||'asset');
      return new File([blob],filename,{type:blob.type});
    };
    const files=[await fetchFile(modelPath)];
    if(logPath)files.push(await fetchFile(logPath));
    await handleAssetFiles(files);
  }catch(error){
    console.error(error);setStatus(`Automatic Colab asset loading failed: ${error.message||error}`);
    document.querySelector('#modelStatus').textContent=`Automatic load failed: ${error.message||error}`;
  }
}
async function startViewer(){await loadProjectOrigin();await loadAssetsFromQuery();}
resize(); animate();
startViewer();
