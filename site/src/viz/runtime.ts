// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

/**
 * Loads the docs visuals in this folder and mounts every [data-viz] element on the page.
 * Each module is a plain script that calls SteelViz.register(name, (root, lib) => ...); _lib.js defines
 * SteelViz and the shared helpers, series.json holds the measured series the figures plot.
 */
import * as THREE from 'three';
import './viz.css';
import './_lib.js';
import series from './series.json';

declare global {
	interface Window {
		THREE: typeof THREE;
		SteelViz: { mountAll: (scope: ParentNode, data: unknown) => void };
	}
}

window.THREE = THREE;
// Every other module registers itself on import; _lib.js above must come first.
await Promise.all(Object.values(import.meta.glob(['./*.js', '!./_lib.js'])).map((load) => load()));
window.SteelViz.mountAll(document, series);
