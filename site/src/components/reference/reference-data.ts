// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import data from '../../data/reference.json';

/** The generated SolverSteel reference (site/scripts/gen_reference.py writes src/data/reference.json). */
export const reference = data;

export type Option = (typeof data.options)[number];

const labels = Object.keys(data.options[0]?.available ?? {});
const release = labels[0];

/** "main" when the option is in a later ref than the release, "branch" when only on the integration branch. */
export function availability(o: Option): 'release' | 'main' | 'branch' {
	const avail = o.available as Record<string, boolean>;
	if (!release || avail[release]) return 'release';
	return labels.some((l) => avail[l]) ? 'main' : 'branch';
}

export const categories = [...new Set(data.options.map((o) => o.category))];

/** The option's glyph from src/viz/glyphs.js (a browser script), evaluated at build time. */
export const glyph: (o: Option) => string = (() => {
	const code = readFileSync(join(process.cwd(), 'src', 'viz', 'glyphs.js'), 'utf8');
	const win: { SteelVizGlyph?: (o: Option) => string } = {};
	new Function('window', code)(win);
	return win.SteelVizGlyph ?? (() => '');
})();
