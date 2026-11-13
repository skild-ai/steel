// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Build-time access to the agent workflows: agents/{README.md,workflows/<name>/README.md,steps/<name>/README.md}.
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, posix } from 'node:path';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import { visit } from 'unist-util-visit';
import { repoRoot } from './tutorials-data.mjs';
import { rehypeTables } from './rehype-tables.mjs';

export const agentsDir = join(repoRoot, 'agents');
const blobUrl = 'https://github.com/skild-ai/steel/blob/main';
const treeUrl = 'https://github.com/skild-ai/steel/tree/main';

let cached = null;

/** The workflows and steps in folder order, each with its summary lines and its README's repository path. */
export function loadWorkflows() {
	if (cached) return cached;
	const workflows = readDocs('workflows', ['Goal', 'Use when', 'Done when']);
	const steps = readDocs('steps', ['Does', 'Use in', 'Passes when']);
	const stepNames = new Set(steps.map((s) => s.name));
	for (const w of workflows) {
		w.steps = [...new Set([...w.text.matchAll(/\]\(\.\.\/\.\.\/steps\/([a-z0-9-]+)\/README\.md\)/g)].map((m) => m[1]))];
		const unknown = w.steps.filter((s) => !stepNames.has(s));
		if (unknown.length) throw new Error(`${w.path}: links to missing steps ${unknown.join(', ')}`);
		w.extras = readdirSync(join(agentsDir, 'workflows', w.name))
			.filter((f) => f.endsWith('.md') && f !== 'README.md')
			.sort()
			.map((f) => readDoc(`agents/workflows/${w.name}/${f}`, f.replace(/\.md$/, '')));
	}
	for (const s of steps) s.usedBy = workflows.filter((w) => w.steps.includes(s.name)).map((w) => w.name);
	cached = { workflows, steps, catalog: readCatalog(workflows) };
	return cached;
}

/** A README rendered to HTML with its title dropped and its links pointed at site pages or at GitHub. */
export async function renderDoc(doc, base) {
	const prefix = base.replace(/\/$/, '');
	const markdown = await createMarkdownProcessor({
		syntaxHighlight: false,
		remarkPlugins: [[remarkSiteLinks, { from: doc.path, prefix }]],
		rehypePlugins: [rehypeTables],
	});
	const body = doc.text.replace(/^# .*\n+/, '').replace(/^(> \*\*[^*]+:\*\* .*\n)+\n?/m, '');
	return (await markdown.render(body)).code;
}

/** Sidebar group: the overview, one link per workflow, the steps, and how to add a workflow. */
export function workflowsSidebar() {
	const { workflows, steps } = loadWorkflows();
	return {
		label: 'Agent workflows',
		items: [
			{ label: 'Overview', slug: 'workflows' },
			...workflows.map((w) => ({ label: w.title, link: `/workflows/${w.name}/` })),
			{
				label: 'Steps',
				collapsed: true,
				items: [{ label: 'All steps', link: '/workflows/steps/' }, ...steps.map((s) => ({ label: s.title, link: `/workflows/steps/${s.name}/` }))],
			},
			{ label: 'Add a workflow', slug: 'workflows/contributing' },
		],
	};
}

/** The site page for a repository path under agents/, or null when the file has none. */
export function sitePath(repoPath) {
	if (repoPath === 'agents/README.md') return '/workflows/';
	const m = repoPath.match(/^agents\/(workflows|steps)\/([a-z0-9-]+)\/([A-Za-z0-9_-]+)\.md$/);
	if (!m) return null;
	const [, kind, name, file] = m;
	const root = kind === 'workflows' ? `/workflows/${name}/` : `/workflows/steps/${name}/`;
	return file === 'README' ? root : kind === 'workflows' ? `${root}${file}/` : null;
}

function readDocs(kind, keys) {
	const dir = join(agentsDir, kind);
	return readdirSync(dir)
		.filter((name) => existsSync(join(dir, name, 'README.md')))
		.sort()
		.map((name) => {
			const doc = readDoc(`agents/${kind}/${name}/README.md`, name);
			doc.summary = Object.fromEntries(keys.map((k) => [k, summaryLine(doc, k)]));
			return doc;
		});
}

function readDoc(path, name) {
	const text = readFileSync(join(repoRoot, path), 'utf8');
	const title = text.match(/^# (.+)$/m)?.[1].trim();
	if (!title) throw new Error(`${path} has no # title`);
	return { name, path, title, text };
}

function summaryLine(doc, key) {
	const m = doc.text.match(new RegExp(`^> \\*\\*${key}:\\*\\* (.+)$`, 'm'));
	if (!m) throw new Error(`${doc.path} has no '> **${key}:**' summary line`);
	return m[1].trim();
}

// The "What are you trying to do?" rows of agents/README.md: [what you want, workflow name].
function readCatalog(workflows) {
	const text = readFileSync(join(agentsDir, 'README.md'), 'utf8');
	const section = text.split(/^## What are you trying to do\?\s*$/m)[1]?.split(/^## /m)[0] ?? '';
	const rows = [...section.matchAll(/^\| (.+?) \| \[([a-z0-9-]+)\]\(workflows\/[a-z0-9-]+\/README\.md\) \|$/gm)].map((m) => ({ want: m[1], name: m[2] }));
	const names = new Set(workflows.map((w) => w.name));
	if (!rows.length || rows.some((r) => !names.has(r.name)) || workflows.some((w) => !rows.some((r) => r.name === w.name))) {
		throw new Error('agents/README.md: the "What are you trying to do?" table must name every workflow once');
	}
	return rows;
}

// Resolve each relative link against the README's repository path: agent docs go to their site page, the rest to GitHub.
function remarkSiteLinks({ from, prefix }) {
	return (tree) => {
		visit(tree, ['link', 'definition'], (node) => {
			const url = node.url ?? '';
			if (!url || /^(?:[a-z]+:|#|\/)/i.test(url)) return;
			const [path, hash] = url.split('#');
			const target = posix.normalize(posix.join(posix.dirname(from), path));
			const page = sitePath(target);
			if (page) node.url = `${prefix}${page}${hash ? `#${hash}` : ''}`;
			else node.url = `${/\.[a-z]+$/i.test(target) ? blobUrl : treeUrl}/${target.replace(/\/$/, '')}${hash ? `#${hash}` : ''}`;
		});
	};
}

/** A summary line's inline Markdown (code spans and bold) as HTML. */
export function inlineHtml(text) {
	const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
	return escaped.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}
