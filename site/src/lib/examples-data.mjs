// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Build-time access to the examples shelf: examples/shelf/<NN_category>/<name>/{presets.toml,README.md}.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { parse } from 'smol-toml';
import { repoRoot } from './tutorials-data.mjs';

export const shelfDir = join(repoRoot, 'examples', 'shelf');
// Media for each example, served at <base>/examples/<name>/.
export const mediaDir = join(process.cwd(), 'public', 'examples');
export const sourceUrl = 'https://github.com/skild-ai/steel/tree/main/examples/shelf';

const SOLVERS = ['steel', 'xpbd', 'vbd', 'mujoco', 'featherstone', 'semi_implicit', 'feather_pgs', 'kamino', 'implicit_mpm', 'style3d'];
const SETTING_KEYS = ['sim', 'collision', 'solver', 'model', 'entries', 'kind'];
const CATEGORY_LABELS = {
	basics: 'Basics',
	basic_contacts: 'Basic contacts',
	many_contacts: 'Many contacts',
	mesh_sdf_contacts: 'Mesh and SDF contacts',
	mimic_joints: 'Mimic joints',
	looped_linkages: 'Looped linkages',
	looped_linkages_contacts: 'Looped linkages with contacts',
	articulations: 'Articulations',
	thin_shells_and_rods: 'Thin shells and rods',
	coupling: 'Coupling',
};

let cached = null;

/** Every example in shelf order, with its presets, the preset its media shows, and its media files. */
export function loadExamples() {
	if (cached) return cached;
	const examples = [];
	const categories = [];
	for (const dir of readdirSync(shelfDir).sort()) {
		const categoryPath = join(shelfDir, dir);
		if (!/^\d+_/.test(dir) || !statSync(categoryPath).isDirectory()) continue;
		const key = dir.replace(/^\d+_/, '');
		const category = { id: key.replace(/_/g, '-'), dir, label: CATEGORY_LABELS[key] ?? key.replace(/_/g, ' ') };
		categories.push(category);
		for (const name of readdirSync(categoryPath).sort()) {
			const presetsPath = join(categoryPath, name, 'presets.toml');
			if (existsSync(presetsPath)) examples.push(readExample(category, name, presetsPath));
		}
	}
	const seen = new Set();
	for (const e of examples) {
		if (seen.has(e.name)) throw new Error(`examples/shelf: two examples named ${e.name}`);
		seen.add(e.name);
	}
	cached = { examples, categories: categories.filter((c) => examples.some((e) => e.category.id === c.id)) };
	return cached;
}

/** The example's example.py and presets.toml as committed. */
export function exampleFiles(example) {
	const dir = join(shelfDir, example.category.dir, example.name);
	return ['example.py', 'presets.toml'].map((name) => ({ name, code: readFileSync(join(dir, name), 'utf8') }));
}

/** The example's README.md, or null. */
export function exampleReadme(example) {
	const path = join(shelfDir, example.category.dir, example.name, 'README.md');
	return existsSync(path) ? readFileSync(path, 'utf8') : null;
}

function readExample(category, name, presetsPath) {
	const data = parse(readFileSync(presetsPath, 'utf8'));
	const gallery = data.gallery ?? {};
	if (!gallery.short) throw new Error(`${presetsPath}: [gallery] needs short = "..."`);
	const presets = [];
	for (const [key, table] of Object.entries(data)) {
		if (!table || typeof table !== 'object' || ['check', 'gallery'].includes(key)) continue;
		const status = table.status ?? 'untested';
		const placeholder = key !== 'none' && status === 'untested' && !SETTING_KEYS.some((k) => k in table);
		if (placeholder) continue;
		presets.push({
			name: key,
			status,
			solver: SOLVERS.includes(key) ? key : null,
			variantOf: table.kind ?? null,
			checked: table.checked ?? '',
			reason: table.reason ?? '',
			isDefault: key === data.default,
		});
	}
	presets.sort((a, b) => b.isDefault - a.isDefault || rank(a) - rank(b));
	const media = {};
	for (const file of ['poster.jpg', 'preview.mp4', 'clip.mp4']) {
		if (existsSync(join(mediaDir, name, file))) media[file.split('.')[0]] = `examples/${name}/${file}`;
	}
	return {
		name,
		title: gallery.title ?? name[0].toUpperCase() + name.slice(1).replace(/_/g, ' '),
		short: gallery.short,
		category,
		path: `${category.dir}/${name}`,
		source: data.source ?? '',
		defaultPreset: data.default,
		presets,
		featured: featuredPreset(data),
		media,
	};
}

// Steel presets first, then other solvers, coupled presets and variants.
function rank(p) {
	if (p.name === 'steel') return 0;
	if (p.name.split('_').includes('steel') && !p.variantOf) return 1;
	if (p.solver) return 2 + SOLVERS.indexOf(p.solver);
	return 20;
}

/** The preset the media shows: [steel] when it has settings, else a coupled preset that includes Steel. */
function featuredPreset(data) {
	const runs = (t) => t && t.status !== 'unsupported' && SETTING_KEYS.slice(0, 5).some((k) => k in t);
	if (runs(data.steel)) return 'steel';
	const coupled = Object.entries(data).filter(
		([k, t]) => k !== 'steel' && typeof t === 'object' && k.split('_').includes('steel') && t.kind !== 'steel' && runs(t),
	);
	if (!coupled.length) return null;
	return coupled.some(([k]) => k === data.default) ? data.default : coupled[0][0];
}

/** Sidebar group for the examples: the gallery, then one collapsed sublist per category. */
export function examplesSidebar() {
	const { examples, categories } = loadExamples();
	return {
		label: 'Examples',
		items: [
			{ label: 'All examples', link: '/examples/' },
			...categories.map((c) => ({
				label: c.label,
				collapsed: true,
				items: examples.filter((e) => e.category.id === c.id).map((e) => ({ label: e.title, link: `/examples/${e.name}/` })),
			})),
		],
	};
}
