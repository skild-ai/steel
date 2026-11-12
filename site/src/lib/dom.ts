/** Small DOM helpers shared by the demos. */

const SVGNS = 'http://www.w3.org/2000/svg';

/** An SVG element with attributes and optional text. */
export function svgEl(tag: string, attrs: Record<string, string | number>, text?: string): SVGElement {
	const n = document.createElementNS(SVGNS, tag);
	for (const k in attrs) n.setAttribute(k, String(attrs[k]));
	if (text !== undefined) n.textContent = text;
	return n;
}

export const esc = (s: unknown): string =>
	String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

/** Three decimals, or exponent notation for very small or large values. */
export const fmt = (x: number, d = 3): string =>
	x === 0 ? '0' : Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4 ? x.toExponential(1) : x.toFixed(d);

/** Element by id inside `root`; throws when missing so a renamed id fails loudly. */
export function byId<T extends HTMLElement | SVGElement = HTMLElement>(root: ParentNode, id: string): T {
	const n = root.querySelector(`#${id}`);
	if (!n) throw new Error(`missing #${id}`);
	return n as T;
}
