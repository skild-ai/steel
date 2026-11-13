#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Check that every example in examples/shelf has a gallery card, a page and media (or a placeholder), and that
// every media folder and media manifest entry belongs to an example.
// usage: node scripts/check-examples.mjs [dist]
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { loadExamples, mediaDir } from '../src/lib/examples-data.mjs';

const dist = process.argv[2] ?? 'dist';
const LIMITS = { 'poster.jpg': 60_000, 'preview.mp4': 300_000, 'clip.mp4': 600_000 };
const { examples } = loadExamples();
const names = new Set(examples.map((e) => e.name));
const manifest = JSON.parse(readFileSync('src/data/example-media.json', 'utf8')).examples;
const gallery = readFileSync(join(dist, 'examples', 'index.html'), 'utf8');
const errors = [];
const warnings = [];

for (const e of examples) {
	const page = join(dist, 'examples', e.name, 'index.html');
	if (!gallery.includes(`id="${e.name}"`)) errors.push(`${e.name}: no card on the gallery`);
	if (!existsSync(page)) errors.push(`${e.name}: no page at examples/${e.name}/`);
	const missing = Object.keys(LIMITS).filter((f) => !existsSync(join(mediaDir, e.name, f)));
	if (missing.length) {
		if (!readFileSync(page, 'utf8').includes('No clip yet')) errors.push(`${e.name}: media missing (${missing.join(', ')}) and no placeholder`);
		warnings.push(`${e.name}: no ${missing.join(', ')}; the card and page show a placeholder`);
	} else if (manifest[e.name]?.preset !== e.featured) {
		warnings.push(`${e.name}: media show ${manifest[e.name]?.preset ?? 'an unrecorded preset'}, the Steel preset is ${e.featured}`);
	}
	for (const [file, limit] of Object.entries(LIMITS)) {
		const path = join(mediaDir, e.name, file);
		if (existsSync(path) && statSync(path).size > limit) errors.push(`${e.name}/${file}: ${statSync(path).size} bytes, over ${limit}`);
	}
}
for (const dir of existsSync(mediaDir) ? readdirSync(mediaDir) : []) {
	if (!names.has(dir)) errors.push(`public/examples/${dir}: no example named ${dir}`);
}
for (const name of Object.keys(manifest)) {
	if (!names.has(name)) errors.push(`src/data/example-media.json: no example named ${name}`);
}

if (warnings.length) console.warn(warnings.join('\n'));
if (errors.length) {
	console.error(errors.join('\n'));
	console.error(`check-examples: ${errors.length} problem(s) in ${examples.length} examples`);
	process.exit(1);
}
console.log(`check-examples: ${examples.length} examples, each with a card and a page; ${examples.length - warnings.length} with current media.`);
