// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

// Loads every built page in headless Chromium and fails on a figure that did not mount, a console error,
// or a table clipped at phone or desktop width. Usage: node scripts/check-figures.mjs [dist]
// STEEL_CHROME=/path/to/chrome uses that browser instead of `npx playwright-core install chromium`.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, relative, sep } from 'node:path';
import { chromium } from 'playwright-core';

const dist = process.argv[2] ?? 'dist';
const base = (process.env.STEEL_SITE_BASE ?? '/').replace(/\/?$/, '/');
const widths = [1440, 1024, 768, 390];

const pages = walk(dist)
	.filter((f) => f.endsWith(`${sep}index.html`))
	.map((f) => base + relative(dist, f).slice(0, -'index.html'.length).split(sep).join('/'));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.mp4': 'video/mp4', '.webm': 'video/webm', '.woff2': 'font/woff2' };
const server = createServer((req, res) => {
	const path = decodeURIComponent(new URL(req.url, 'http://x').pathname).slice(base.length - 1);
	try {
		const file = join(dist, path.endsWith('/') ? `${path}index.html` : path);
		res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' }).end(readFileSync(file));
	} catch {
		res.writeHead(404).end();
	}
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ executablePath: process.env.STEEL_CHROME || undefined });
const failures = [];
try {
	for (const width of widths) {
		const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
		for (const path of pages) {
			for (const problem of await checkPage(context, path, width)) failures.push(`${path} @${width}px: ${problem}`);
		}
		await context.close();
	}
} finally {
	await browser.close();
	server.close();
}

if (failures.length) {
	console.error(failures.join('\n'));
	console.error(`check-figures: ${failures.length} problem(s) in ${pages.length} pages.`);
	process.exit(1);
}
console.log(`check-figures: ${pages.length} pages at ${widths.join(', ')} px, every figure mounted, no console errors.`);

async function checkPage(context, path, width) {
	const page = await context.newPage();
	const problems = [];
	page.on('console', (m) => m.type() === 'error' && problems.push(`console error: ${m.text()}`));
	page.on('pageerror', (e) => problems.push(`page error: ${e.message}`));
	page.on('requestfailed', (r) => problems.push(`request failed: ${r.url()}`));
	const response = await page.goto(origin + path, { waitUntil: 'load' });
	if (!response?.ok()) problems.push(`HTTP ${response?.status()}`);
	// Visuals mount from a module script after top-level awaits; wait until every one is marked.
	await page.waitForFunction(() => [...document.querySelectorAll('[data-viz]')].every((n) => n.dataset.vizState), null, { timeout: 15000 }).catch(() => {});
	await page.waitForTimeout(200);
	problems.push(
		...(await page.evaluate(() => {
			const out = [];
			const visible = (n) => {
				const r = n.getBoundingClientRect();
				return r.width > 0 && r.height > 0;
			};
			for (const root of document.querySelectorAll('[data-viz]')) {
				const name = root.dataset.viz + (root.dataset.term ? `:${root.dataset.term}` : '');
				if (root.dataset.vizState !== 'mounted') out.push(`visual "${name}" is ${root.dataset.vizState ?? 'not mounted'}`);
				else if (root.matches('figure') && ![...root.children].some((c) => c.tagName !== 'FIGCAPTION' && visible(c)))
					out.push(`visual "${name}" mounted but drew nothing visible`);
			}
			for (const frame of document.querySelectorAll('figure .frame')) {
				if (!frame.querySelector('svg, canvas, img, video') || !visible(frame)) out.push(`figure "${frame.closest('figure').id || frame.closest('figure').querySelector('figcaption')?.textContent.slice(0, 40)}" is empty`);
			}
			for (const glyph of document.querySelectorAll('svg.oi')) {
				if (!visible(glyph) || !glyph.children.length) out.push(`option glyph in ${glyph.closest('[id]')?.id} is empty`);
			}
			for (const table of document.querySelectorAll('.sl-markdown-content table, .not-content table')) {
				let scroller = table.parentElement;
				while (scroller && getComputedStyle(scroller).overflowX === 'visible') scroller = scroller.parentElement;
				const clip = scroller ?? document.documentElement;
				const scrollable = ['auto', 'scroll'].includes(getComputedStyle(clip).overflowX);
				if (table.getBoundingClientRect().width > clip.clientWidth + 1 && !scrollable)
					out.push(`table "${table.querySelector('th')?.textContent}" is clipped`);
			}
			if (document.documentElement.scrollWidth > window.innerWidth + 1) out.push(`page scrolls sideways (${document.documentElement.scrollWidth}px)`);
			return out;
		})),
	);
	await page.close();
	return problems;
}

function walk(dir) {
	return readdirSync(dir).flatMap((name) => {
		const path = join(dir, name);
		return statSync(path).isDirectory() ? walk(path) : [path];
	});
}
