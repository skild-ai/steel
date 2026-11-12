// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Build-time access to the repository's tutorials/ folder: index.toml, <script>.toml, <script>.py, recordings/.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parse } from 'smol-toml';

// npm scripts run from site/; STEEL_REPO_ROOT overrides for other layouts.
export const repoRoot = resolve(process.env.STEEL_REPO_ROOT ?? join(process.cwd(), '..'));
export const tutorialsDir = join(repoRoot, 'tutorials');

let cached = null;

/** Every tutorial in page order, with its metadata, script name and group. */
export function loadTutorials() {
	if (cached) return cached;
	const index = parse(readFileSync(join(tutorialsDir, 'index.toml'), 'utf8'));
	const byId = new Map();
	for (const file of readdirSync(tutorialsDir)) {
		if (!file.endsWith('.toml') || file === 'index.toml' || file === 'ruff.toml') continue;
		const meta = parse(readFileSync(join(tutorialsDir, file), 'utf8'));
		const script = file.slice(0, -'.toml'.length);
		if (byId.has(meta.id)) throw new Error(`tutorials/${file}: duplicate id ${meta.id}`);
		byId.set(meta.id, { ...meta, script, file: `${script}.py` });
	}
	const list = [];
	for (const group of index.group) {
		for (const id of group.tutorials) {
			const meta = byId.get(id);
			if (!meta) throw new Error(`tutorials/index.toml lists ${id}, which has no <script>.toml`);
			list.push({ ...meta, group: group.name, num: list.length + 1 });
			byId.delete(id);
		}
	}
	if (byId.size) throw new Error(`tutorials missing from tutorials/index.toml: ${[...byId.keys()].join(', ')}`);
	cached = {
		tutorials: list,
		groups: index.group.map((g) => g.name).filter((g) => list.some((t) => t.group === g)),
		planned: index.planned ?? [],
	};
	return cached;
}

/** The tutorial's script source exactly as committed. */
export function tutorialSource(meta) {
	return readFileSync(join(tutorialsDir, meta.file), 'utf8');
}

export function replaySource() {
	return readFileSync(join(tutorialsDir, 'replay.py'), 'utf8');
}

/** The recording, or null; `stale` is set when the script changed after it was recorded. */
export function loadRecording(meta) {
	const path = join(tutorialsDir, 'recordings', `${meta.id}.json`);
	if (!existsSync(path)) return null;
	const rec = JSON.parse(readFileSync(path, 'utf8'));
	// Same digest as tutorials/source_hash.py: the script without its leading SPDX header.
	const source = readFileSync(join(tutorialsDir, meta.file), 'latin1').replace(/^(#![^\n]*\n)?(?:# SPDX-[^\n]*\n)+\n?/, '$1');
	const sha = createHash('sha256').update(source, 'latin1').digest('hex');
	rec.stale = rec.script_sha256 !== sha;
	return rec;
}

/** Sections and bullets of tutorials/DR_GAPS.md. */
export function drGaps() {
	const path = join(tutorialsDir, 'DR_GAPS.md');
	if (!existsSync(path)) return [];
	const sections = [];
	let current = null;
	for (const raw of readFileSync(path, 'utf8').split('\n')) {
		const line = raw.trimEnd();
		if (line.startsWith('## ')) sections.push((current = { title: line.slice(3), items: [] }));
		else if (line.startsWith('- ') && current) current.items.push(line.slice(2));
		else if (line.startsWith('  ') && current?.items.length) current.items[current.items.length - 1] += ' ' + line.trim();
	}
	return sections;
}
