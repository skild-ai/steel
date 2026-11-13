// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

// Example clips: a lazy poster until a clip is wanted, then one muted looping video at a time.
// With a pointer that hovers, a card plays while hovered or focused; on touch screens, the card nearest the
// middle of the screen plays. prefers-reduced-motion plays a clip only on an explicit hover, focus or tap.

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const canHover = matchMedia('(hover: hover) and (pointer: fine)');
let playing: HTMLVideoElement | null = null;

/** Wire every [data-clip] element under root. */
export function initClips(root: ParentNode = document) {
	const holders = [...root.querySelectorAll<HTMLElement>('[data-clip]')];
	for (const holder of holders) {
		const card = holder.closest<HTMLElement>('[data-clip-card]') ?? holder;
		card.addEventListener('pointerenter', (e) => e.pointerType === 'mouse' && play(holder));
		card.addEventListener('pointerleave', (e) => e.pointerType === 'mouse' && stop(holder));
		card.addEventListener('focusin', () => play(holder));
		card.addEventListener('focusout', (e) => !card.contains(e.relatedTarget as Node) && stop(holder));
	}
	if (canHover.matches || reduceMotion.matches) return;
	const visible = new Map<HTMLElement, number>();
	const observer = new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				const holder = entry.target as HTMLElement;
				if (entry.intersectionRatio >= 0.75) visible.set(holder, entry.boundingClientRect.top + entry.boundingClientRect.height / 2);
				else {
					visible.delete(holder);
					stop(holder);
				}
			}
			const middle = innerHeight / 2;
			const best = [...visible.entries()].sort((a, b) => Math.abs(a[1] - middle) - Math.abs(b[1] - middle))[0];
			if (best) play(best[0]);
		},
		{ threshold: [0, 0.75, 1] },
	);
	for (const holder of holders) observer.observe(holder);
}

function play(holder: HTMLElement) {
	let video = holder.querySelector('video');
	if (!video) {
		video = document.createElement('video');
		Object.assign(video, { muted: true, loop: true, playsInline: true, preload: 'none' });
		video.setAttribute('aria-hidden', 'true');
		video.src = holder.dataset.clip!;
		holder.append(video);
	}
	if (playing && playing !== video) pause(playing);
	playing = video;
	video.play().then(
		() => holder.classList.add('is-playing'),
		() => {},
	);
}

function stop(holder: HTMLElement) {
	const video = holder.querySelector('video');
	if (video) pause(video);
}

function pause(video: HTMLVideoElement) {
	video.pause();
	video.parentElement?.classList.remove('is-playing');
	if (playing === video) playing = null;
}
