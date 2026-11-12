// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

import { visit } from 'unist-util-visit';

/** Prefix the site base to root-relative links in Markdown, so content can link to `/concepts/...`. */
export function remarkBaseLinks({ base = '/' } = {}) {
	const prefix = base.replace(/\/$/, '');
	return (tree) => {
		if (!prefix) return;
		visit(tree, ['link', 'definition'], (node) => {
			if (node.url?.startsWith('/') && !node.url.startsWith('//') && !node.url.startsWith(prefix + '/')) {
				node.url = prefix + node.url;
			}
		});
	};
}
