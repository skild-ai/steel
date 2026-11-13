// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Checks in headless Chromium that example clips behave: the page clip plays on screen, holds still under reduced
// motion, keeps a seek made before its metadata loads, and changes speed; gallery previews play one at a time.
// Usage: node scripts/check-clips.mjs [dist]  (STEEL_CHROME as in check-figures)
import { readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { chromium } from 'playwright-core';

const dist = process.argv[2] ?? 'dist';
const base = (process.env.STEEL_SITE_BASE ?? '/').replace(/\/?$/, '/');
const example = 'air_hockey';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.mp4': 'video/mp4', '.woff2': 'font/woff2' };
// Serves byte ranges, as static hosts do, so the browser can seek before a clip has fully loaded.
const server = createServer((req, res) => {
	const path = decodeURIComponent(new URL(req.url, 'http://x').pathname).slice(base.length - 1);
	const file = join(dist, path.endsWith('/') ? `${path}index.html` : path);
	let size;
	try {
		size = statSync(file).size;
	} catch {
		return res.writeHead(404).end();
	}
	const headers = { 'content-type': types[extname(file)] ?? 'application/octet-stream', 'accept-ranges': 'bytes' };
	const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? '');
	if (!range) return res.writeHead(200, headers).end(readFileSync(file));
	const start = range[1] ? Number(range[1]) : size - Number(range[2]);
	const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
	res.writeHead(206, { ...headers, 'content-range': `bytes ${start}-${end}/${size}`, 'content-length': end - start + 1 });
	res.end(readFileSync(file).subarray(start, end + 1));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}${base}`;
const browser = await chromium.launch({ executablePath: process.env.STEEL_CHROME || undefined });
const failures = [];
const check = (ok, what) => (ok ? console.log(`  ok  ${what}`) : failures.push(what));
const video = (page) => page.evaluate(() => {
	const v = document.querySelector('.clip-player video');
	return { t: v.currentTime, d: v.duration, paused: v.paused, rate: v.playbackRate, scrub: Number(document.querySelector('.clip-player .scrub').value) };
});

try {
	const reduced = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
	let page = await reduced.newPage();
	page.on('pageerror', (e) => failures.push(`page error: ${e.message}`));
	await page.goto(`${origin}examples/${example}/`);
	await page.waitForTimeout(1000);
	check((await video(page)).paused, 'reduced motion: the page clip holds still');
	await page.locator('.clip-player .scrub').evaluate((el) => {
		el.value = '500';
		el.dispatchEvent(new Event('input', { bubbles: true }));
	});
	await page.waitForFunction(() => document.querySelector('.clip-player video').readyState >= 1, null, { timeout: 15000 });
	await page.waitForTimeout(500);
	let v = await video(page);
	check(Math.abs(v.t - v.d / 2) < 0.2 && v.paused && Math.abs(v.scrub - 500) <= 20, `a seek before metadata lands mid-clip and stays paused (t=${v.t.toFixed(2)} of ${v.d})`);
	await reduced.close();

	const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
	page = await context.newPage();
	page.on('pageerror', (e) => failures.push(`page error: ${e.message}`));
	await page.goto(`${origin}examples/${example}/`);
	await page.waitForFunction(() => document.querySelector('.clip-player video').currentTime > 0.5, null, { timeout: 15000 }).catch(() => {});
	v = await video(page);
	check(!v.paused && v.t > 0.5, `the page clip plays on screen (t=${v.t.toFixed(2)})`);
	await page.click('.clip-player .speed [data-s="2"]');
	v = await video(page);
	const pressed = await page.getAttribute('.clip-player .speed [data-s="2"]', 'aria-pressed');
	check(v.rate === 2 && pressed === 'true', '2x sets the playback rate and is announced as pressed');

	await page.goto(`${origin}examples/`);
	const cards = page.locator('.example-card');
	await cards.nth(0).hover();
	await page.waitForFunction(() => document.querySelectorAll('.clip.is-playing').length === 1, null, { timeout: 15000 }).catch(() => {});
	await cards.nth(1).hover();
	await page.waitForTimeout(1500);
	const playing = await page.evaluate(() => [...document.querySelectorAll('.clip video')].filter((x) => !x.paused).length);
	check(playing === 1, `gallery: hovering a second card leaves one preview playing (${playing})`);
	await context.close();
} finally {
	await browser.close();
	server.close();
}

if (failures.length) {
	console.error(failures.map((f) => `FAIL ${f}`).join('\n'));
	process.exit(1);
}
console.log('check-clips: example clips play, seek, change speed and play one at a time.');
