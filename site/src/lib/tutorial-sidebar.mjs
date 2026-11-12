// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

import { loadTutorials } from './tutorials-data.mjs';

/** Sidebar group for the tutorials: the index, then one collapsed sublist per tutorial group. */
export function tutorialSidebar() {
	const { tutorials, groups } = loadTutorials();
	return {
		label: 'Tutorials',
		items: [
			{ label: 'All tutorials', link: '/tutorials/' },
			...groups.map((group) => ({
				label: group,
				collapsed: true,
				items: tutorials
					.filter((t) => t.group === group)
					.map((t) => ({ label: `${String(t.num).padStart(2, '0')} ${t.title}`, link: `/tutorials/${t.id}/` })),
			})),
		],
	};
}
