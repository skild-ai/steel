// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

import { visit, SKIP } from 'unist-util-visit';

const wbr = () => ({ type: 'element', tagName: 'wbr', properties: {}, children: [] });
const text = (node) => (node.type === 'text' ? node.value : (node.children ?? []).map(text).join(''));

/**
 * Wrap every Markdown table in a scroll box, label each cell with its column header for the stacked phone layout,
 * and let inline code in cells break after _ . : / = , ( and inside camelCase.
 */
export function rehypeTables() {
	return (tree) => {
		visit(tree, 'element', (node, index, parent) => {
			if (node.tagName !== 'table' || !parent || index === undefined) return;
			const rows = [];
			visit(node, 'element', (n) => {
				if (n.tagName === 'tr') rows.push(n);
			});
			const cells = (tr) => tr.children.filter((c) => c.type === 'element' && (c.tagName === 'td' || c.tagName === 'th'));
			const headers = rows.length ? cells(rows[0]).map((c) => text(c).trim()) : [];
			for (const tr of rows.slice(1)) cells(tr).forEach((c, i) => (c.properties.dataLabel = headers[i] ?? ''));
			visit(node, 'element', (code) => {
				if (code.tagName !== 'code') return;
				code.children = code.children.flatMap((c) =>
					c.type === 'text' ? c.value.split(/(?<=[_.:/=,(]|[a-z](?=[A-Z]))(?=.)/).flatMap((part, i) => (i ? [wbr(), { type: 'text', value: part }] : [{ type: 'text', value: part }])) : [c],
				);
			});
			if (headers.length >= 3) node.properties.className = [...(node.properties.className ?? []), 'stack'];
			parent.children[index] = {
				type: 'element',
				tagName: 'div',
				properties: { className: ['table-scroll'], tabIndex: 0, role: 'region', ariaLabel: `Table: ${headers.join(', ')}` },
				children: [node],
			};
			return SKIP;
		});
	};
}
