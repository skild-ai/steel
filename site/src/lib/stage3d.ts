/**
 * Shared three.js helpers for the 3D stages on the site: the Steel studio look (sky gradient, checker
 * floor, hemisphere fill and warm key light, ACES), theme tokens, and a small orbit stage for demos.
 * Colors always come from CSS tokens, read at draw time so light and dark themes both work.
 */
import * as THREE from 'three';

export { THREE };

/** A CSS custom property of the root element, such as `--series-2`. */
export const css = (name: string): string =>
	getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** A three.js color from a CSS token. */
export const col = (name: string): THREE.Color => new THREE.Color(css(name));

/** Call `cb` whenever the site theme or the system color scheme changes. */
export function onThemeChange(cb: () => void): void {
	const mq = window.matchMedia('(prefers-color-scheme: dark)');
	mq.addEventListener?.('change', cb);
	new MutationObserver(cb).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

/** Vertical sky gradient from `--stage-top` (zenith) to `--stage` (horizon). */
export function skyTexture(): THREE.CanvasTexture {
	const c = document.createElement('canvas');
	c.width = 2;
	c.height = 256;
	const g = c.getContext('2d')!;
	const grad = g.createLinearGradient(0, 0, 0, 256);
	grad.addColorStop(0, css('--stage-top') || css('--stage'));
	grad.addColorStop(1, css('--stage'));
	g.fillStyle = grad;
	g.fillRect(0, 0, 2, 256);
	const tex = new THREE.CanvasTexture(c);
	tex.colorSpace = THREE.SRGBColorSpace;
	return tex;
}

/** Checker floor of `--floor` and `--floor-alt` tiles; repeat it to set the tile size. */
export function checkerTexture(): THREE.CanvasTexture {
	const c = document.createElement('canvas');
	c.width = c.height = 128;
	const g = c.getContext('2d')!;
	g.fillStyle = css('--floor');
	g.fillRect(0, 0, 128, 128);
	g.fillStyle = css('--floor-alt') || css('--floor');
	g.fillRect(0, 0, 64, 64);
	g.fillRect(64, 64, 64, 64);
	const tex = new THREE.CanvasTexture(c);
	tex.colorSpace = THREE.SRGBColorSpace;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.anisotropy = 8;
	return tex;
}

/** A WebGL renderer with the studio tone mapping, or null when WebGL is unavailable. */
export function studioRenderer(opts: THREE.WebGLRendererParameters = {}): THREE.WebGLRenderer | null {
	let renderer: THREE.WebGLRenderer;
	try {
		renderer = new THREE.WebGLRenderer({ antialias: true, ...opts });
	} catch {
		return null;
	}
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.0;
	renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
	return renderer;
}

/** Cool hemisphere fill plus a warm key light from above; returns the key light. */
export function studioLights(scene: THREE.Scene): THREE.DirectionalLight {
	// Intensities are the legacy (non-physical) values times pi, so scenes keep their brightness.
	const hemi = new THREE.HemisphereLight(0xc9d3df, 0x8f857a, 0.65 * Math.PI);
	hemi.position.set(0, 0, 1);
	scene.add(hemi);
	const fill = new THREE.DirectionalLight(0xdfe8f5, 0.18 * Math.PI);
	fill.position.set(-4, 3, 2);
	scene.add(fill);
	const sun = new THREE.DirectionalLight(0xfff2db, 1.05 * Math.PI);
	sun.position.set(3, -4, 6);
	scene.add(sun);
	scene.add(sun.target);
	return sun;
}

export interface StageView {
	target: [number, number, number];
	dist: number;
	theta: number;
	phi: number;
}

export interface Stage {
	group: THREE.Group;
	req: () => void;
	clear: () => void;
	resize: () => void;
}

/**
 * A small orbit stage inside `el` (a `.stage3d` element): drag or arrow keys orbit, wheel zooms, a `.reset`
 * button restores `view`. Renders on demand. Returns null, with a note in `el`, without WebGL.
 */
export function makeStage(el: HTMLElement, view: StageView): Stage | null {
	const renderer = studioRenderer();
	if (!renderer) {
		el.insertAdjacentHTML('beforeend', '<p class="nogl">The 3D view needs WebGL, which this browser did not provide.</p>');
		return null;
	}
	el.insertBefore(renderer.domElement, el.firstChild);
	const scene = new THREE.Scene();
	const cam = new THREE.PerspectiveCamera(35, 1, 0.01, 200);
	cam.up.set(0, 0, 1);
	studioLights(scene);
	const group = new THREE.Group();
	scene.add(group);
	const st = { ...view };
	const target = new THREE.Vector3(...view.target);
	let queued = false;
	const draw = () => {
		queued = false;
		renderer.render(scene, cam);
	};
	const req = () => {
		if (!queued) {
			queued = true;
			requestAnimationFrame(draw);
		}
	};
	const place = () => {
		const c = Math.cos(st.phi);
		cam.position.set(
			target.x + st.dist * c * Math.cos(st.theta),
			target.y + st.dist * c * Math.sin(st.theta),
			target.z + st.dist * Math.sin(st.phi),
		);
		cam.lookAt(target);
		req();
	};
	const resize = () => {
		const w = el.clientWidth;
		const h = el.clientHeight;
		if (!w || !h) return;
		renderer.setSize(w, h, false);
		cam.aspect = w / h;
		cam.updateProjectionMatrix();
		req();
	};
	new ResizeObserver(resize).observe(el);
	let drag: [number, number] | null = null;
	el.addEventListener('pointerdown', (e) => {
		if ((e.target as HTMLElement).closest('.reset')) return;
		drag = [e.clientX, e.clientY];
		el.setPointerCapture(e.pointerId);
	});
	el.addEventListener('pointermove', (e) => {
		if (!drag) return;
		st.theta -= (e.clientX - drag[0]) * 0.01;
		st.phi = Math.max(-1.2, Math.min(1.45, st.phi + (e.clientY - drag[1]) * 0.01));
		drag = [e.clientX, e.clientY];
		place();
	});
	const end = () => (drag = null);
	el.addEventListener('pointerup', end);
	el.addEventListener('pointercancel', end);
	el.addEventListener(
		'wheel',
		(e) => {
			e.preventDefault();
			st.dist = Math.max(view.dist * 0.3, Math.min(view.dist * 3, st.dist * Math.exp(e.deltaY * 0.001)));
			place();
		},
		{ passive: false },
	);
	el.addEventListener('keydown', (e) => {
		const d = ({ ArrowLeft: [0.08, 0], ArrowRight: [-0.08, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] } as Record<string, number[]>)[e.key];
		if (!d) return;
		e.preventDefault();
		st.theta += d[0];
		st.phi = Math.max(-1.2, Math.min(1.45, st.phi + d[1]));
		place();
	});
	el.querySelector('.reset')?.addEventListener('click', () => {
		Object.assign(st, view);
		place();
	});
	const clear = () => {
		group.traverse((o) => {
			const m = o as THREE.Mesh;
			m.geometry?.dispose();
			(m.material as THREE.Material | undefined)?.dispose?.();
		});
		group.clear();
		(scene.background as THREE.Texture | null)?.dispose?.();
		scene.background = skyTexture();
	};
	resize();
	place();
	return { group, req, clear, resize };
}

type V3 = [number, number, number];

export function line3(points: V3[], color: THREE.Color, dashed = false): THREE.Line {
	const g = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
	const m = dashed
		? new THREE.LineDashedMaterial({ color, dashSize: 0.06, gapSize: 0.05 })
		: new THREE.LineBasicMaterial({ color });
	const l = new THREE.Line(g, m);
	if (dashed) l.computeLineDistances();
	return l;
}

export function sphere3(p: V3, r: number, color: THREE.Color): THREE.Mesh {
	const m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), new THREE.MeshStandardMaterial({ color, roughness: 0.45 }));
	m.position.set(...p);
	return m;
}

export function arrow3(from: V3, vec: V3, color: THREE.Color, head?: number): THREE.Object3D {
	const v = new THREE.Vector3(...vec);
	const len = v.length();
	if (len < 1e-6) return new THREE.Group();
	const h = head || Math.min(0.3 * len, 0.06);
	return new THREE.ArrowHelper(v.normalize(), new THREE.Vector3(...from), len, color, h, h * 0.6);
}

export function boxEdges(size: V3, pos: V3, color: THREE.Color, fill?: THREE.Color, opacity = 1): THREE.Group {
	const g = new THREE.BoxGeometry(...size);
	const grp = new THREE.Group();
	if (fill !== undefined) {
		grp.add(
			new THREE.Mesh(
				g,
				new THREE.MeshStandardMaterial({ color: fill, roughness: 0.45, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 }),
			),
		);
	}
	grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color })));
	grp.position.set(...pos);
	return grp;
}
