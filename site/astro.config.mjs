// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';
import { remarkBaseLinks } from './src/lib/remark-base-links.mjs';
import { rehypeTables } from './src/lib/rehype-tables.mjs';
import { tutorialSidebar } from './src/lib/tutorial-sidebar.mjs';

// Deploy under a subpath with STEEL_SITE_BASE=/steel/docs; absolute links in content get the prefix.
const base = process.env.STEEL_SITE_BASE ?? '/';
// The "Inside Steel" post is hosted separately; point the sidebar link at it.
const insideSteelUrl = process.env.INSIDE_STEEL_URL ?? '/inside-steel/';

export default defineConfig({
	vite: { server: { fs: { allow: ['..'] } } },
	// Host names other than localhost that `npm run dev` / `npm run preview` answer, e.g. STEEL_ALLOWED_HOSTS=myhost.
	server: { allowedHosts: (process.env.STEEL_ALLOWED_HOSTS ?? '').split(',').filter(Boolean) },
	site: process.env.STEEL_SITE_URL ?? 'https://skild-ai.github.io',
	base,
	trailingSlash: 'always',
	markdown: {
		processor: unified({ remarkPlugins: [remarkMath, [remarkBaseLinks, { base }]], rehypePlugins: [rehypeKatex, rehypeTables] }),
	},
	integrations: [
		starlight({
			title: 'Steel',
			description:
				'Documentation and tutorials for Steel (Scalable Throughput-optimized Engine for Embodied Learning), the GPU rigid-body and articulation solver in Newton (newton.solvers.SolverSteel).',
			logo: { light: './src/assets/skild-mark-light.svg', dark: './src/assets/skild-mark-dark.svg', alt: 'Skild AI', replacesTitle: false },
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/skild-ai/steel' }],
			editLink: { baseUrl: 'https://github.com/skild-ai/steel/edit/main/site/' },
			components: { Footer: './src/components/Footer.astro' },
			customCss: ['katex/dist/katex.min.css', './src/styles/steel.css'],
			head: [
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap',
					},
				},
			],
			expressiveCode: {
				themes: ['github-light', 'github-dark'],
				styleOverrides: {
					borderRadius: '4px',
					borderColor: 'var(--rule)',
					codeBackground: 'var(--panel)',
					codeFontFamily: 'var(--mono)',
					uiFontFamily: 'var(--mono)',
					frames: {
						editorTabBarBackground: 'var(--panel)',
						editorActiveTabBackground: 'var(--bg)',
						editorActiveTabIndicatorTopColor: 'var(--hot-fill)',
						terminalBackground: 'var(--panel)',
						terminalTitlebarBackground: 'var(--panel)',
						frameBoxShadowCssValue: 'none',
					},
				},
			},
			sidebar: [
				{
					label: 'Start',
					items: [
						{ label: 'Overview', slug: 'start/overview' },
						{ label: 'Versions and features', slug: 'start/versions' },
						{ label: 'Installation', slug: 'start/installation' },
						{ label: 'Quick start', slug: 'start/quick-start' },
					],
				},
				tutorialSidebar(),
				{ label: 'Concepts', items: [{ autogenerate: { directory: 'concepts' } }] },
				{ label: 'Guides', items: [{ autogenerate: { directory: 'guides' } }] },
				{ label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
				{
					label: 'More',
					items: [
						{ label: 'FAQ', slug: 'faq' },
						{ label: 'Changelog', slug: 'changelog' },
						{ label: 'Inside Steel', link: insideSteelUrl },
						{ label: 'Contributing', slug: 'contributing' },
					],
				},
			],
			plugins: [
				starlightLinksValidator({
					errorOnRelativeLinks: false,
					// Tutorial pages and the notices file are generated outside the docs collection; scripts/check-links.mjs checks them.
					exclude: [`${base}tutorials/`, `${base}tutorials/**`, `${base}inside-steel/`, `${base}third-party-notices.txt`],
				}),
			],
		}),
	],
});
