#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Write dist/third-party-notices.txt for the third-party code and fonts the built site ships. Fails when a
// file in dist/_astro has a name that is neither a site/src module nor a component's pattern below, or when a
// shipped component has no license text. Names are a tripwire, not a dependency analysis: review the list when
// client imports change. usage: node scripts/write-notices.mjs [dist]
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const dist = process.argv[2] ?? 'dist';
const astroFiles = readdirSync(join(dist, '_astro'));

// files: which dist/_astro files come from the component. license: its license text.
const components = [
	{ pkg: 'three', license: 'node_modules/three/LICENSE', files: /^three\.module\./ },
	{
		pkg: 'katex',
		license: 'node_modules/katex/LICENSE',
		files: /^middleware\..*\.css$/,
		note: 'KaTeX CSS, bundled into middleware.*.css. The KaTeX fonts are listed under the SIL Open Font License below.',
	},
	{ pkg: 'astro', license: 'node_modules/astro/LICENSE', files: /^page\..*\.js$/, note: 'Client scripts (page.*.js).' },
	{ pkg: 'vite', license: 'node_modules/vite/LICENSE.md', licenseUntil: '# Licenses of bundled dependencies', files: /^preload-helper\./, note: 'Module preload helper.' },
	{
		pkg: '@astrojs/starlight',
		license: 'node_modules/@astrojs/starlight/LICENSE',
		files: /^(print\.|middleware\.|_id_\.|starlight-toc\.|(Search|TableOfContents|MobileTableOfContents)\.astro_)/,
		note: 'Theme styles and scripts (print, middleware, _id_ styles; search and table-of-contents scripts).',
	},
	{ pkg: 'astro-expressive-code', license: 'node_modules/astro-expressive-code/LICENSE', files: /^ec\./, note: 'Code block styles and scripts.' },
	{ pkg: 'pagefind', license: 'node_modules/pagefind/LICENSE/LICENSE', shipped: existsSync(join(dist, 'pagefind')), note: 'Search index and search scripts in pagefind/.' },
	{ pkg: '@pagefind/default-ui', license: 'node_modules/pagefind/LICENSE/LICENSE', files: /^ui-core\./, note: 'Search interface; part of the Pagefind project.' },
	{ name: 'Svelte runtime, compiled into @pagefind/default-ui', license: 'licenses/svelte-LICENSE.md', files: /^ui-core\./ },
];
const katexFont = /^KaTeX_.*\.(woff2?|ttf)$/;

// First-party chunks are named after a module in site/src: a viz module, a lib module or a component's script.
const sourceNames = new Set(
	['src/viz', 'src/lib', 'src/components', 'src/components/demos', 'src/components/figures', 'src/components/tutorials', 'src/assets'].flatMap((d) =>
		readdirSync(d).map((f) => f.replace(/\.(js|ts|mjs|astro|svg)$/, '')),
	),
);
const firstParty = (f) => sourceNames.has(f.split(/[.]/)[0]);
const unknown = astroFiles.filter((f) => !katexFont.test(f) && !firstParty(f) && !components.some((c) => c.files?.test(f)));
if (unknown.length) throw new Error(`No license source for shipped files: ${unknown.join(', ')}. Add them to scripts/write-notices.mjs.`);

const rule = '-'.repeat(78);
const sections = [];
for (const c of components) {
	if (!(c.shipped ?? astroFiles.some((f) => c.files.test(f)))) continue;
	if (!existsSync(c.license)) throw new Error(`${c.pkg ?? c.name} ships but ${c.license} is missing.`);
	let text = readFileSync(c.license, 'utf8');
	if (c.licenseUntil) {
		const end = text.indexOf(c.licenseUntil);
		if (end < 0) throw new Error(`${c.license} no longer has "${c.licenseUntil}"; update scripts/write-notices.mjs.`);
		text = text.slice(0, end);
	}
	const title = c.pkg ? `${c.pkg} ${JSON.parse(readFileSync(`node_modules/${c.pkg}/package.json`, 'utf8')).version}` : c.name;
	sections.push([rule, title, ...(c.note ? [c.note] : []), '', text.trim()].join('\n'));
}

const fonts = astroFiles.filter((f) => katexFont.test(f));
if (fonts.length) {
	const families = [...new Set(fonts.map((f) => f.split('-')[0]))].sort();
	sections.push(
		[
			rule,
			`KaTeX fonts (${families.length} families, ${fonts.length} files in _astro/)`,
			'',
			'Copyright (c) 2009-2010, Design Science, Inc. (<www.mathjax.org>)',
			'Copyright (c) 2014-2018 Khan Academy (<www.khanacademy.org>),',
			`with Reserved Font Names ${families.join(', ')}.`,
			'',
			'This Font Software is licensed under the SIL Open Font License, Version 1.1.',
			'',
			readFileSync('licenses/OFL-1.1.txt', 'utf8').trim(),
		].join('\n'),
	);
}

const header = [
	'Third-party notices for the Steel documentation site',
	'',
	'The site is built from https://github.com/skild-ai/steel and is licensed under Apache-2.0. It ships the',
	'third-party software and fonts below, each under its own license. The Geist fonts are loaded from Google',
	'Fonts and are not part of this site. "Skild", "Skild AI" and the Skild logo are trademarks of Skild AI and',
	'are not licensed under Apache-2.0.',
	'',
].join('\n');
writeFileSync(join(dist, 'third-party-notices.txt'), `${header}\n${sections.join('\n\n')}\n`);
console.log(`Wrote third-party-notices.txt: ${sections.length} components, ${fonts.length} font files.`);
