/**
 * The tutorial replay: loads tutorials/recordings/<id>.json, rebuilds every recorded variant's shapes,
 * plays the body poses back with a scrubber, and draws the metric charts with a playhead.
 */
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { THREE, css, checkerTexture, onThemeChange, skyTexture, studioLights, studioRenderer } from '../stage3d';
import { esc } from '../dom';

interface Shape { b: number; t: number; s: number[]; x: number[]; v?: number[]; i?: number[] }
interface Particles { count: number; frames: string; radius: number; tris: number[] | null; X?: Float32Array }
interface Variant {
	label: string; fps: number; ghost?: boolean; slot?: number | null; scene?: string | null; bodies: number; frames: number;
	poses: string; shapes: Shape[]; metric: Record<string, number | null>[]; awake?: number[][] | null; particles?: Particles | null;
	P?: Float32Array; U?: Uint32Array;
}
interface Chart { key?: string; label: string; unit?: string; multi?: string[]; alt?: { key: string; label: string; unit?: string }; log?: boolean; scene?: string }
export interface ReplayMeta { metric?: Chart & { charts?: Chart[] }; spacing?: number[]; fit?: string; speed?: number }
interface Doc extends ReplayMeta { variants: Variant[] }
interface Live {
	v: Variant; root: THREE.Group; bodies: THREE.Group[]; bodyMats: Record<number, [THREE.MeshStandardMaterial | THREE.MeshBasicMaterial, THREE.Color][]>;
	col: string; sleepCol: THREE.Color; slot: number; cloth: THREE.Mesh | THREE.Points | null; pair?: Live;
}

const VCOL = () => [css('--series-1'), css('--series-2'), css('--series-3'), css('--series-4')];

function decode(b64: string): Float32Array {
	const bin = atob(b64);
	const buf = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
	return new Float32Array(buf.buffer);
}

// Newton GeoType ids.
function geomFor(sh: Shape): THREE.BufferGeometry | null {
	const s = sh.s;
	switch (sh.t) {
		case 3: return new THREE.SphereGeometry(s[0], 32, 20);
		case 4: { const g = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20); g.rotateX(Math.PI / 2); return g; }
		case 5: { const g = new THREE.SphereGeometry(1, 28, 18); g.scale(s[0], s[1], s[2]); return g; }
		case 6: { const g = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 36); g.rotateX(Math.PI / 2); return g; }
		case 7: return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
		case 8: case 10: {
			if (!sh.v || !sh.i) return null;
			const g = new THREE.BufferGeometry();
			g.setAttribute('position', new THREE.Float32BufferAttribute(sh.v, 3));
			g.setIndex(sh.i);
			g.scale(s[0], s[1], s[2]);
			const flat = g.toNonIndexed();
			flat.computeVertexNormals();
			return flat;
		}
		case 9: { const g = new THREE.ConeGeometry(s[0], 2 * s[1], 32); g.rotateX(Math.PI / 2); return g; }
	}
	return null;
}

export async function mountReplay(root: HTMLElement): Promise<void> {
	const q = <T extends HTMLElement = HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
	const stage = q('.stage');
	const status = q('.status');
	const readout = q('.readout');
	const chartwrap = q('.chartwrap');
	const playBtn = q<HTMLButtonElement>('.play');
	const scrub = q<HTMLInputElement>('.scrub');
	const timeEl = q('.time');
	const meta: ReplayMeta = JSON.parse(root.dataset.meta || '{}');

	const renderer = studioRenderer({ preserveDrawingBuffer: true });
	if (!renderer) {
		stage.insertAdjacentHTML('beforeend', '<p class="nogl">The replay needs WebGL, which this browser did not provide.</p>');
		return;
	}
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFSoftShadowMap;
	stage.insertBefore(renderer.domElement, stage.firstChild);
	// Vertical swipes scroll the page on touch screens; sideways drags orbit.
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(38, 2, 0.01, 200);
	camera.up.set(0, 0, 1);
	const controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	renderer.domElement.style.touchAction = 'pan-y';
	const sun = studioLights(scene);
	sun.castShadow = true;
	sun.shadow.mapSize.set(2048, 2048);
	let world = new THREE.Group();
	scene.add(world);

	let doc: Doc;
	try {
		const r = await fetch(root.dataset.src!);
		if (!r.ok) throw new Error(`HTTP ${r.status}`);
		doc = { ...meta, ...(await r.json()) };
	} catch (e) {
		status.textContent = `The recording could not load (${(e as Error).message}). Reload the page to try again.`;
		return;
	}
	doc.variants.forEach((v) => {
		v.P = decode(v.poses);
		if (v.particles) v.particles.X = decode(v.particles.frames);
	});

	let variants: Live[] = [];
	let t = 0, T = 1, playing = true, speed = doc.speed || 1, last = performance.now();
	let fitBox = new THREE.Box3();

	const slotOf = (v: Variant, k: number) => (v.slot !== null && v.slot !== undefined ? v.slot : k);
	const variantOffset = (v: Variant, k: number) => {
		const sp = doc.spacing || [0, 0.6, 0];
		const n = new Set(doc.variants.map((x, i) => slotOf(x, i))).size;
		const c = slotOf(v, k) - (n - 1) / 2;
		return new THREE.Vector3(sp[0] * c, sp[1] * c, sp[2] * c);
	};

	function build(keepCamera: boolean) {
		world.traverse((o) => {
			const m = o as THREE.Mesh;
			m.geometry?.dispose();
			(m.material as THREE.Material | undefined)?.dispose?.();
		});
		scene.remove(world);
		world = new THREE.Group();
		scene.add(world);
		(scene.background as THREE.Texture | null)?.dispose?.();
		scene.background = skyTexture();
		const cols = VCOL();
		const staticCol = new THREE.Color(css('--static'));
		const sleepCol = new THREE.Color(css('--sleep'));
		fitBox = new THREE.Box3();
		let groundZ: number | null = null;
		variants = [];
		// Within each scene the last run is the subject (green); the others are blue, orange and violet.
		const compare = [cols[0], cols[2], cols[3]];
		const sceneSlots: Record<string, number[]> = {};
		doc.variants.forEach((x, i) => {
			const key = x.scene || '';
			(sceneSlots[key] ||= []);
			if (!sceneSlots[key].includes(slotOf(x, i))) sceneSlots[key].push(slotOf(x, i));
		});
		const colorOf = (slot: number, sc?: string | null) => {
			const list = sceneSlots[sc || ''] || [slot];
			return slot === list[list.length - 1] ? cols[1] : compare[list.indexOf(slot) % 3];
		};
		const diverge = doc.variants.some((x) => x.ghost);
		doc.variants.forEach((v, k) => {
			const slot = slotOf(v, k);
			// With sleep flags, awake bodies are green so blue-gray can mean asleep.
			const col = new THREE.Color(v.awake ? cols[1] : diverge ? cols[0] : colorOf(slot, v.scene));
			const vroot = new THREE.Group();
			vroot.position.copy(variantOffset(v, k));
			world.add(vroot);
			const bodies: THREE.Group[] = [];
			for (let b = 0; b < v.bodies; b++) {
				const g = new THREE.Group();
				vroot.add(g);
				bodies.push(g);
			}
			const bodyMats: Live['bodyMats'] = {};
			v.shapes.forEach((sh) => {
				if (sh.t === 1) {
					groundZ = sh.x[2];
					return;
				}
				const isStatic = sh.b < 0;
				if (v.ghost && isStatic) return;
				const geo = geomFor(sh);
				if (!geo) return;
				const shade = isStatic ? staticCol.clone() : col.clone().offsetHSL(0, 0, ((sh.b * 0.137) % 0.08) - 0.04);
				const mat = v.ghost
					? new THREE.MeshBasicMaterial({ color: new THREE.Color(css('--fg')), wireframe: true, transparent: true, opacity: 0.55 })
					: new THREE.MeshStandardMaterial({ color: shade, roughness: 0.45, metalness: 0.05, side: sh.t === 8 ? THREE.DoubleSide : THREE.FrontSide });
				const mesh = new THREE.Mesh(geo, mat);
				mesh.castShadow = !v.ghost;
				mesh.receiveShadow = true;
				mesh.position.set(sh.x[0], sh.x[1], sh.x[2]);
				mesh.quaternion.set(sh.x[3], sh.x[4], sh.x[5], sh.x[6]);
				if (isStatic) vroot.add(mesh);
				else {
					bodies[sh.b].add(mesh);
					(bodyMats[sh.b] ||= []).push([mat, shade.clone()]);
				}
			});
			// Awake-flag runs chart in green unless their scene compares several runs.
			const key = v.awake && (sceneSlots[v.scene || ''] || []).length < 2 ? cols[1] : colorOf(slot, v.scene);
			// Deformables from a coupled solver: a triangle mesh when given, else points.
			let cloth: Live['cloth'] = null;
			if (v.particles) {
				const n = v.particles.count;
				const geo = new THREE.BufferGeometry();
				geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
				if (v.particles.tris) {
					geo.setIndex(v.particles.tris);
					cloth = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(cols[2]), roughness: 0.8, side: THREE.DoubleSide }));
					cloth.castShadow = cloth.receiveShadow = true;
				} else {
					cloth = new THREE.Points(geo, new THREE.PointsMaterial({ color: new THREE.Color(cols[2]), size: 2 * (v.particles.radius || 0.01) }));
				}
				cloth.frustumCulled = false;
				vroot.add(cloth);
			}
			variants.push({ v, root: vroot, bodies, bodyMats, col: '#' + new THREE.Color(key).getHexString(), sleepCol, slot, cloth });
		});
		// Pair each run with its ghost rerun to show per-body divergence.
		variants.forEach((V) => {
			if (!V.v.ghost) return;
			const A = variants.find((W) => !W.v.ghost && W.slot === V.slot);
			if (A) A.pair = V;
		});
		readout.hidden = !variants.some((V) => V.pair);
		if (groundZ !== null) {
			const size = 30;
			const tex = checkerTexture(); // 0.5 m tiles
			tex.repeat.set(size, size);
			const g = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshStandardMaterial({ map: tex, roughness: 1, metalness: 0 }));
			g.position.z = groundZ - 0.0005;
			g.receiveShadow = true;
			world.add(g);
		}
		T = Math.max(...doc.variants.map((v) => (v.frames - 1) / v.fps));
		// Frame the motion: the start pose plus later poses, leaving out bodies that fly far away.
		setPose(doc.fit === 'end' ? T : 0);
		world.updateMatrixWorld(true);
		variants.forEach((V) => V.root.traverse((o) => (o as THREE.Mesh).isMesh && fitBox.expandByObject(o)));
		const c0 = fitBox.getCenter(new THREE.Vector3());
		const reach = 0.4 + fitBox.getSize(new THREE.Vector3()).length();
		(doc.fit === 'end' ? [] : [T / 3, (2 * T) / 3, T]).forEach((time) => {
			setPose(time);
			world.updateMatrixWorld(true);
			variants.forEach((V) =>
				V.root.traverse((o) => {
					if (!(o as THREE.Mesh).isMesh) return;
					const b = new THREE.Box3().setFromObject(o);
					if (b.getCenter(new THREE.Vector3()).distanceTo(c0) < reach) fitBox.union(b);
				}),
			);
		});
		setPose(Math.min(t, T));
		if (!keepCamera) fitCamera();
		buildLabels();
		drawCharts();
	}

	function fitCamera() {
		const c = fitBox.getCenter(new THREE.Vector3());
		const s = fitBox.getSize(new THREE.Vector3());
		const sp = doc.spacing || [0, 0.6, 0];
		const alongY = Math.abs(sp[1]) >= Math.abs(sp[0]) && !(sp[0] === 0 && sp[1] === 0 && s.x > s.y * 1.3);
		const width = alongY ? s.y : s.x, depth = alongY ? s.x : s.y;
		const aspect = camera.aspect || 2;
		const halfW = Math.max(0.25, width * 0.55), halfH = Math.max(0.2, Math.max(s.z, depth * 0.5) * 0.62);
		const fovV = (camera.fov * Math.PI) / 180;
		const dist = Math.max(halfH / Math.tan(fovV / 2), halfW / (Math.tan(fovV / 2) * aspect)) * 1.15 + depth * 0.5;
		const r = Math.max(halfW, halfH);
		// A higher view looks into containers such as the determinism bins.
		const lift = doc.fit === 'end' ? 1.1 : 0.42;
		const dir = (alongY ? new THREE.Vector3(-1.0, -0.2, lift) : new THREE.Vector3(0.2, -1.0, lift)).normalize();
		camera.position.copy(c).addScaledVector(dir, dist);
		camera.near = dist / 200;
		camera.far = dist * 40;
		camera.updateProjectionMatrix();
		controls.target.copy(c);
		controls.update();
		sun.target.position.copy(c);
		sun.position.copy(c).add(new THREE.Vector3(-0.4, -0.25, 0.85).multiplyScalar(7 * r));
		scene.fog = new THREE.Fog(new THREE.Color(css('--stage')), dist * 6, dist * 25);
		const sc = sun.shadow.camera;
		const e = r * 2.4;
		sc.left = -e; sc.right = e; sc.top = e; sc.bottom = -e; sc.near = 0.1; sc.far = 20 * r + 10;
		sc.updateProjectionMatrix();
	}

	function setPose(time: number) {
		t = time;
		variants.forEach((V) => {
			const v = V.v;
			const f = Math.min(v.frames - 1, Math.max(0, time * v.fps));
			const f0 = Math.floor(f), f1 = Math.min(v.frames - 1, f0 + 1), a = f - f0;
			const B = v.bodies, P = v.P!;
			for (let b = 0; b < B; b++) {
				const i0 = (f0 * B + b) * 7, i1 = (f1 * B + b) * 7;
				const g = V.bodies[b];
				g.position.set(P[i0] + (P[i1] - P[i0]) * a, P[i0 + 1] + (P[i1 + 1] - P[i0 + 1]) * a, P[i0 + 2] + (P[i1 + 2] - P[i0 + 2]) * a);
				const q0 = new THREE.Quaternion(P[i0 + 3], P[i0 + 4], P[i0 + 5], P[i0 + 6]);
				g.quaternion.copy(q0.slerp(new THREE.Quaternion(P[i1 + 3], P[i1 + 4], P[i1 + 5], P[i1 + 6]), a));
			}
			if (V.cloth && v.particles?.X) {
				const X = v.particles.X, n = v.particles.count;
				const attr = V.cloth.geometry.attributes.position as THREE.BufferAttribute;
				const pos = attr.array as Float32Array;
				const i0 = f0 * n * 3, i1 = f1 * n * 3;
				for (let k = 0; k < n * 3; k++) pos[k] = X[i0 + k] + (X[i1 + k] - X[i0 + k]) * a;
				attr.needsUpdate = true;
				if ((V.cloth as THREE.Mesh).isMesh) V.cloth.geometry.computeVertexNormals();
			}
			if (v.awake) {
				const aw = v.awake[Math.min(v.awake.length - 1, Math.round(f))] || [];
				for (const b in V.bodyMats) V.bodyMats[b].forEach(([m, base]) => m.color.copy(aw[+b] === 0 ? V.sleepCol : base));
			}
		});
		const pairs = variants.filter((V) => V.pair);
		if (pairs.length) {
			const hot = new THREE.Color(css('--hot-fill'));
			readout.innerHTML = pairs
				.map((V) => {
					const A = V.v, Bv = V.pair!.v;
					const fr = Math.min(A.frames - 1, Bv.frames - 1, Math.round(time * A.fps));
					let worst = 0, identical = true;
					const AU = A.U || (A.U = new Uint32Array(A.P!.buffer));
					const BU = Bv.U || (Bv.U = new Uint32Array(Bv.P!.buffer));
					for (let b = 0; b < A.bodies; b++) {
						const i = (fr * A.bodies + b) * 7;
						let same = true;
						for (let k = 0; k < 7; k++) if (AU[i + k] !== BU[i + k]) same = false;
						const d = Math.hypot(A.P![i] - Bv.P![i], A.P![i + 1] - Bv.P![i + 1], A.P![i + 2] - Bv.P![i + 2]) * 1000;
						identical = identical && same;
						worst = Math.max(worst, d);
						// Neutral when identical, green as the divergence grows from 1 µm to 10 cm.
						const al = same ? 0 : Math.min(1, Math.max(0.15, (Math.log10(Math.max(d, 1e-3)) + 3) / 5));
						(V.bodyMats[b] || []).forEach(([m, base]) => m.color.copy(base).lerp(hot, al));
					}
					const label = A.label.replace(/, run \d+$/, '');
					return `<div><span>${esc(label)}</span>${identical ? '<b>0</b> · bitwise identical' : `max divergence <b>${worst >= 10 ? worst.toFixed(0) : worst.toPrecision(2)} mm</b>`}</div>`;
				})
				.join('');
		}
		timeEl.textContent = `${time.toFixed(time < 1 ? 3 : 2)} s / ${T.toFixed(2)} s`;
		scrub.value = String(Math.round((1000 * time) / T));
		root.querySelectorAll<SVGLineElement>('.playhead').forEach((l) => {
			const span = +l.dataset.t! || T;
			const x = +l.dataset.x0! + +l.dataset.w! * Math.min(1, time / span);
			l.setAttribute('x1', String(x));
			l.setAttribute('x2', String(x));
		});
		const aw = variants.length && variants[0].v.awake ? variants[0].v : null;
		if (aw) {
			const m = aw.metric[Math.min(aw.metric.length - 1, Math.round(time * aw.fps))];
			status.textContent = m ? `awake bodies: ${m.awake_bodies} of ${aw.bodies} · blue is asleep` : '';
		} else status.textContent = '';
	}

	const labels = q('.labels');
	function buildLabels() {
		labels.innerHTML = variants
			.map((V, i) =>
				V.v.ghost ? '' : `<div class="vlabel" data-i="${i}"><i style="background:${V.col}"></i>${esc(V.pair ? V.v.label.replace(/, run 1$/, ' (solid A, wireframe B)') : V.v.label)}</div>`,
			)
			.join('');
	}
	function placeLabels() {
		const w = stage.clientWidth, h = stage.clientHeight;
		const placed: { x: number; y: number; w: number }[] = [];
		labels.querySelectorAll<HTMLElement>('.vlabel').forEach((el) => {
			const V = variants[+el.dataset.i!];
			if (!V) return;
			const box = new THREE.Box3();
			V.root.traverse((o) => (o as THREE.Mesh).isMesh && box.expandByObject(o));
			if (box.isEmpty()) return;
			const p = new THREE.Vector3((box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, Math.max(box.min.z, -0.05)).project(camera);
			const lw = el.offsetWidth || 120;
			const x = Math.max(lw / 2 + 4, Math.min(w - lw / 2 - 4, ((p.x + 1) / 2) * w));
			let y = Math.max(30, ((1 - p.y) / 2) * h + 10);
			// Push down past labels already placed so none overlap.
			for (let guard = 0; guard < 20 && placed.some((r) => Math.abs(r.x - x) < (r.w + lw) / 2 + 4 && Math.abs(r.y - y) < 24); guard++) y += 24;
			placed.push({ x, y, w: lw });
			el.style.left = x + 'px';
			el.style.top = Math.min(h - 26, y) + 'px';
		});
	}

	// ---------- charts
	const series = (v: Variant, key: string) => v.metric.map((m) => (m && m[key] !== undefined ? m[key] : null));
	function nice(lo: number, hi: number, n: number) {
		const step0 = (hi - lo) / n;
		const mag = Math.pow(10, Math.floor(Math.log10(step0)));
		const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
		const out: number[] = [];
		for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
		return out;
	}
	const fmt = (v: number) => {
		const a = Math.abs(v);
		return a >= 10 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : a === 0 ? '0' : v.toPrecision(2);
	};
	const fmt3 = (v: number | null | undefined) => {
		if (v === undefined || v === null || !isFinite(v)) return '–';
		const a = Math.abs(v);
		return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : a === 0 ? '0' : v.toPrecision(3);
	};
	interface Line { d: (number | null)[]; c: string; fps: number; dash?: boolean }
	function chartSVG(title: string, lines: Line[], small: boolean, log?: boolean) {
		// Each chart spans its own runs' duration; the width follows the container so text stays legible.
		const tspan = Math.max(...lines.map((L) => (L.d.length - 1) / L.fps), 1e-6);
		const W = Math.max(320, Math.min(900, chartwrap.clientWidth || 900));
		const H = small ? 200 : W < 600 ? 210 : 230;
		const pad = { l: 58, r: 16, t: 12, b: 28 };
		// Log scale: values become decades, and exact zeros sit on a marked floor one decade below the smallest value.
		let floor = 0;
		if (log) {
			const pos = lines.flatMap((L) => L.d.filter((x): x is number => x !== null && x > 0));
			floor = Math.floor(Math.log10(pos.length ? Math.min(...pos) : 1e-3)) - 1;
			lines = lines.map((L) => ({ ...L, d: L.d.map((x) => (x === null ? null : x > 0 ? Math.log10(x) : floor)) }));
		}
		const ys = lines.flatMap((L) => L.d.filter((x): x is number => x !== null && isFinite(x)));
		if (!ys.length) return '';
		let lo = Math.min(...ys), hi = Math.max(...ys);
		if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
		const span = hi - lo;
		lo -= span * 0.06;
		hi += span * 0.06;
		const X = (s: number) => pad.l + (W - pad.l - pad.r) * (s / tspan);
		const Y = (y: number) => pad.t + (H - pad.t - pad.b) * (1 - (y - lo) / (hi - lo));
		let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
		const ticks = log ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => Math.ceil(lo) + i) : nice(lo, hi, 4);
		const tickLabel = (v: number) => (!log ? fmt(v) : v === floor ? 'exact 0' : v >= 0 ? String(10 ** v) : '1e' + v);
		ticks.forEach((v) => {
			s += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--rule)" stroke-width="1" ${log && v === floor ? 'stroke-dasharray="3 3"' : ''}/><text x="${pad.l - 8}" y="${Y(v) + 4}" text-anchor="end" font-family="var(--mono)" font-size="12" fill="var(--muted)">${tickLabel(v)}</text>`;
		});
		nice(0, tspan, W < 600 ? 4 : 6).forEach((v) => {
			if (v <= tspan + 1e-9) s += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle" font-family="var(--mono)" font-size="12" fill="var(--muted)">${fmt(v)} s</text>`;
		});
		lines.forEach((L) => {
			let d = '', pen = false;
			L.d.forEach((y, i) => {
				if (y === null || !isFinite(y)) { pen = false; return; }
				d += (pen ? 'L' : 'M') + X(i / L.fps).toFixed(1) + ',' + Y(y).toFixed(1);
				pen = true;
			});
			s += `<path d="${d}" fill="none" stroke="${L.c}" stroke-width="2" ${L.dash ? 'stroke-dasharray="6 4"' : ''} stroke-linejoin="round"/>`;
		});
		s += `<line class="playhead" data-t="${tspan}" data-x0="${pad.l}" data-w="${W - pad.l - pad.r}" x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${H - pad.b}" stroke="var(--hot-fill)" stroke-width="1.5"/></svg>`;
		return s;
	}
	function chartBlock(M: Chart) {
		const vis = variants.filter((V) => !V.v.ghost && (!M.scene || V.v.scene === M.scene));
		const rowsOf = variants.filter((V) => (!V.v.ghost || doc.variants.some((x) => x.ghost)) && (!M.scene || V.v.scene === M.scene));
		const multi = M.multi && M.multi.length > 1;
		let html = '';
		if (multi) {
			const kc = VCOL();
			html = vis
				.map((V) => `<div>${vis.length > 1 ? `<div class="ct">${esc(V.v.label)}</div>` : ''}${chartSVG(V.v.label, M.multi!.map((k, j) => ({ d: series(V.v, k), c: kc[j % 4], fps: V.v.fps, dash: k === 'chain_weight' })), vis.length > 1)}</div>`)
				.join('');
			html += `<div class="legend">${M.multi!.map((k, j) => `<span><i style="background:${kc[j % 4]}"></i>${esc(k)}</span>`).join('')}</div>`;
		} else {
			html = `<div>${chartSVG(M.label, vis.map((V) => ({ d: series(V.v, M.key!), c: V.col, fps: V.v.fps })), false, M.log)}</div>`;
		}
		if (M.alt) {
			html += `<div><div class="ct">${esc(M.alt.label)}${M.alt.unit ? ' [' + esc(M.alt.unit) + ']' : ''}</div>${chartSVG(M.alt.label, vis.map((V) => ({ d: series(V.v, M.alt!.key), c: V.col, fps: V.v.fps })), true)}</div>`;
		}
		const keys = M.multi || [M.key!];
		const head = `<tr><th>Variant</th>${keys.map((k) => `<th>final ${esc(M.multi ? k : M.label)}${M.unit ? ' [' + esc(M.unit) + ']' : ''}</th>`).join('')}${keys.length === 1 ? '<th>min … max</th>' : ''}</tr>`;
		const rows = rowsOf.map((V) => {
			const v = V.v;
			const fin = v.metric[v.metric.length - 1] || {};
			let cells = keys.map((k) => `<td class="num">${fmt3(fin[k])}</td>`).join('');
			if (keys.length === 1) {
				const sr = series(v, keys[0]).filter((x): x is number => x !== null);
				cells += `<td class="num">${fmt3(Math.min(...sr))} … ${fmt3(Math.max(...sr))}</td>`;
			}
			return `<tr><td><i style="background:${V.col}"></i>${esc(v.label)}</td>${cells}</tr>`;
		});
		return (
			`<div class="chartblock"><div class="charth"><span>${esc(M.label)}</span><span>${M.unit ? '[' + esc(M.unit) + ']' : ''}</span></div>` +
			`<div class="charts${multi && vis.length > 1 ? ' multi' : ''}">${html}</div>` +
			`<div class="tbl"><table><thead>${head}</thead><tbody>${rows.join('')}</tbody></table></div></div>`
		);
	}
	function drawCharts() {
		const M = doc.metric;
		const specs = M ? M.charts || [M] : [];
		chartwrap.hidden = !specs.length;
		chartwrap.innerHTML = specs.map(chartBlock).join('');
		setPose(t);
	}

	// ---------- controls
	const setPlaying = (p: boolean) => {
		playing = p;
		playBtn.textContent = p ? 'Pause' : t >= T - 1e-6 ? 'Replay' : 'Play';
	};
	playBtn.addEventListener('click', () => {
		if (!playing && t >= T - 1e-6) t = 0;
		setPlaying(!playing);
	});
	scrub.addEventListener('input', () => {
		setPlaying(false);
		setPose((T * +scrub.value) / 1000);
	});
	const speedGroup = q('.speed');
	const setSpeed = (s: number) => {
		speed = s;
		speedGroup.querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
			b.classList.toggle('on', +b.dataset.s! === s);
			b.setAttribute('aria-pressed', String(+b.dataset.s! === s));
		});
	};
	speedGroup.addEventListener('click', (e) => {
		const b = (e.target as HTMLElement).closest('button');
		if (b) setSpeed(+b.dataset.s!);
	});
	q('.recenter').addEventListener('click', () => fitCamera());

	const resize = () => {
		const w = stage.clientWidth, h = stage.clientHeight;
		if (!w || !h) return;
		renderer.setSize(w, h, false);
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
	};
	new ResizeObserver(resize).observe(stage);
	let chartWidth = chartwrap.clientWidth;
	new ResizeObserver(() => {
		if (Math.abs(chartwrap.clientWidth - chartWidth) > 24) {
			chartWidth = chartwrap.clientWidth;
			drawCharts();
		}
	}).observe(chartwrap);
	// Render only while the stage is on screen.
	let visible = true;
	new IntersectionObserver((entries) => (visible = entries[0].isIntersecting)).observe(stage);

	resize();
	build(false);
	setSpeed(speed);
	setPlaying(true);
	onThemeChange(() => build(true));

	function tick(now: number) {
		const dt = Math.min(0.1, (now - last) / 1000);
		last = now;
		if (playing) {
			let nt = t + dt * speed;
			if (nt > T) {
				nt = T;
				t = T;
				setPlaying(false);
			}
			setPose(nt);
		}
		if (visible) {
			controls.update();
			renderer.render(scene, camera);
			placeLabels();
		}
		requestAnimationFrame(tick);
	}
	requestAnimationFrame(tick);
}
