// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Checks in headless Chromium that tutorial replays start from 0 s once loaded and visible: on a fresh load, after
// sidebar navigation and back, behind a slow recording, below the fold, after a stalled tab, and that animated
// figures restart on a tab switch. Usage: node scripts/check-replay.mjs [dist]  (STEEL_CHROME as in check-figures)
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { chromium } from 'playwright-core';

const dist = process.argv[2] ?? 'dist';
const base = (process.env.STEEL_SITE_BASE ?? '/').replace(/\/?$/, '/');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.woff2': 'font/woff2' };
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
const origin = `http://127.0.0.1:${server.address().port}${base}`;
const browser = await chromium.launch({ executablePath: process.env.STEEL_CHROME || undefined });
const failures = [];
const check = (ok, what) => (ok ? console.log(`  ok  ${what}`) : failures.push(what));

try {
	const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
	const page = await context.newPage();
	page.on('pageerror', (e) => failures.push(`page error: ${e.message}`));

	await page.goto(origin + 'tutorials/hello-steel/');
	await expectStart(page, 'fresh load of hello-steel');

	await page.locator('nav.sidebar a[href$="/tutorials/gyroscopic/"]').evaluate((a) => a.click());
	await page.waitForURL(/\/tutorials\/gyroscopic\/$/);
	await expectStart(page, 'sidebar navigation to gyroscopic');

	await page.goBack();
	await expectStart(page, 'back to hello-steel');
	await page.goForward();
	await expectStart(page, 'forward to gyroscopic');

	// The recording arrives 3 s after the page; the clock starts only once it is drawn.
	await page.route('**/recordings/joint-limits.json', async (route) => {
		await new Promise((r) => setTimeout(r, 3000));
		await route.continue();
	});
	await page.goto(origin + 'tutorials/joint-limits/');
	await expectStart(page, 'slow recording load');
	await page.unrouteAll();

	// A tab that gets no frames for 3 s (backgrounded or stalled) resumes where it was instead of skipping ahead.
	await page.goto(origin + 'tutorials/gyroscopic/');
	await expectStart(page, 'gyroscopic before a stall');
	const before = await replayTime(page);
	await page.evaluate(() => {
		const end = performance.now() + 3000;
		while (performance.now() < end);
	});
	await page.waitForTimeout(300);
	const after = await replayTime(page);
	check(after - before < 1.5, `a 3 s stall moved the replay ${(after - before).toFixed(2)} s (< 1.5 s)`);

	// When playback ends the button offers a replay from 0.
	await page.goto(origin + 'tutorials/speculative-contacts/');
	await page.waitForFunction(() => document.querySelector('.replay .play')?.textContent === 'Replay', null, { timeout: 10000 }).catch(() => {});
	check((await page.textContent('.replay .play')) === 'Replay', 'a finished replay shows a Replay button');
	await page.click('.replay .play');
	check((await replayTime(page)) < 0.2, 'Replay restarts from 0 s');

	// Below the fold the clock holds at 0 until the stage scrolls into view.
	const short = await context.newPage();
	await short.setViewportSize({ width: 1440, height: 160 });
	await short.goto(origin + 'tutorials/hello-steel/');
	await short.waitForFunction(() => document.querySelector('.replay .time')?.textContent.includes('/'));
	await short.waitForTimeout(2000);
	check((await replayTime(short)) === 0, 'an offscreen replay holds at 0 s');
	await short.locator('.replay .stage').scrollIntoViewIfNeeded();
	await expectStart(short, 'the same replay scrolled into view');
	await short.close();

	// An animated figure restarts when its view tab switches.
	await page.goto(origin + 'concepts/reduced-coordinates/');
	await page.waitForFunction(() => document.querySelector('[data-viz=reduced]')?.dataset.vizState === 'mounted');
	await page.locator('[data-viz=reduced]').scrollIntoViewIfNeeded();
	await page.waitForTimeout(1500);
	const stretch = () => page.evaluate(() => document.querySelector('[data-viz=reduced] svg text:last-of-type')?.textContent);
	const mid = await stretch();
	const restarted = await page.evaluate(() => {
		const [d2, d3] = document.querySelectorAll('[data-viz=reduced] [data-role=view] button');
		d3.click();
		d2.click();
		return document.querySelector('[data-viz=reduced] svg text:last-of-type')?.textContent;
	});
	check(mid !== restarted && /now 0\.0 mm/.test(restarted ?? ''), `reduced figure restarts on a view switch (${mid} -> ${restarted})`);
	await context.close();
} finally {
	await browser.close();
	server.close();
}

if (failures.length) {
	console.error(failures.join('\n'));
	console.error(`check-replay: ${failures.length} problem(s).`);
	process.exit(1);
}
console.log('check-replay: replays start from 0 s on every entry and advance.');

async function replayTime(page) {
	return page.evaluate(() => parseFloat(document.querySelector('.replay .time').textContent));
}

// The recording loads, the readout starts near 0 s, and playback advances.
async function expectStart(page, what) {
	await page.waitForFunction(() => document.querySelector('.replay .time')?.textContent.includes('/'), null, { timeout: 15000 });
	const t0 = await replayTime(page);
	await page.waitForTimeout(400);
	const t1 = await replayTime(page);
	check(t0 < 0.5 && t1 > t0 && (await page.textContent('.replay .play')) === 'Pause', `${what}: starts at ${t0.toFixed(3)} s and advances to ${t1.toFixed(3)} s`);
}
