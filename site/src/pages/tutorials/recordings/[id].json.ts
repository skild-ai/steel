import type { APIRoute, GetStaticPaths } from 'astro';
import { loadRecording, loadTutorials } from '../../../lib/tutorials-data.mjs';

/** Serves tutorials/recordings/<id>.json as a static file next to the tutorial pages. */
export const getStaticPaths: GetStaticPaths = () =>
	loadTutorials()
		.tutorials.filter((t) => loadRecording(t))
		.map((t) => ({ params: { id: t.id }, props: { meta: t } }));

export const GET: APIRoute = ({ props }) => {
	const { stale, ...rec } = loadRecording(props.meta);
	return new Response(JSON.stringify(rec), { headers: { 'Content-Type': 'application/json' } });
};
