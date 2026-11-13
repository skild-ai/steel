#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Check every internal link and #anchor in the built site (dist/), including generated tutorial pages.
// usage: node scripts/check-links.mjs [dist]
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, posix, relative } from 'node:path';

const dist = process.argv[2] ?? 'dist';
const base = (process.env.STEEL_SITE_BASE ?? '/').replace(/\/?$/, '/');
const pages = new Map();

for (const file of walk(dist)) {
	if (!file.endsWith('.html')) continue;
	const html = readFileSync(file, 'utf8');
	const ids = new Set([...html.matchAll(/\sid=(?:"([^"]+)"|'([^']+)')/g)].map((m) => decode(m[1] ?? m[2])));
	const links = [...html.matchAll(/\s(?:href|src|poster|data-clip)=(?:"([^"]+)"|'([^']+)')/g)].map((m) => decode(m[1] ?? m[2]).replace(/\?[^#]*/, ''));
	pages.set('/' + relative(dist, file).split('\\').join('/'), { ids, links });
}

const errors = [];
for (const [page, { links }] of pages) {
	for (const link of links) {
		if (/^(?:[a-z]+:|\/\/|#$)/i.test(link)) continue;
		const [pathPart, hash] = link.split('#');
		const pageUrl = base + page.slice(1);
		let target = pathPart ? (pathPart.startsWith('/') ? pathPart : posix.join(posix.dirname(pageUrl), pathPart)) : pageUrl;
		if (pathPart.endsWith('/') && !target.endsWith('/')) target += '/';
		if (pathPart && !target.startsWith(base)) {
			errors.push(`${page}: ${link} is outside the site base ${base}`);
			continue;
		}
		target = '/' + target.slice(base.length);
		if (target.endsWith('/')) target += 'index.html';
		const onDisk = join(dist, target);
		if (!existsSync(onDisk) || statSync(onDisk).isDirectory()) {
			errors.push(`${page}: ${link} does not exist`);
			continue;
		}
		if (hash && target.endsWith('.html') && !pages.get(target)?.ids.has(hash)) {
			errors.push(`${page}: ${link} has no #${hash}`);
		}
	}
}

if (errors.length) {
	console.error(errors.join('\n'));
	console.error(`${errors.length} broken internal link(s) in ${pages.size} pages`);
	process.exit(1);
}
console.log(`All internal links and anchors are valid in ${pages.size} pages.`);

function* walk(dir) {
	for (const name of readdirSync(dir)) {
		const path = join(dir, name);
		if (statSync(path).isDirectory()) yield* walk(path);
		else yield path;
	}
}

function decode(s) {
	return s.replace(/&amp;/g, '&').replace(/&#x27;|&#39;/g, "'").replace(/&quot;/g, '"');
}
