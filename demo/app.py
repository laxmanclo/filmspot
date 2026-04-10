from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from api.service import ingest_movie, query_movie


def _ingest(movie_file: str | None, movie_id: str, fps: float, semantic_k: int):
    if not movie_file:
        return "", "Upload a movie file first."

    chosen_id = movie_id.strip() or None
    result = ingest_movie(movie_path=movie_file, movie_id=chosen_id, fps=fps, semantic_k=semantic_k)
    status = (
        f"Indexed **{result['movie_id']}**\\n\\n"
        f"- Nodes: {result['node_count']}\\n"
        f"- Duration: {result['duration_sec']:.2f}s\\n"
        f"- Index: {result['index_dir']}"
    )
    return result["movie_id"], status


def _render_scenes(result: dict) -> str:
    scenes = result.get("scenes", [])
    if not scenes:
        return "No scenes found."

    lines: list[str] = []
    for idx, scene in enumerate(scenes, start=1):
        conflict = " ⚠ conflict" if scene.get("conflict") else ""
        lines.append(
            "\\n".join(
                [
                    f"### Scene {idx}{conflict}",
                    f"- Time: {scene['start_t']:.2f}s → {scene['end_t']:.2f}s",
                    f"- Scores: visual={scene['visual_score']:.3f}, transcript={scene['transcript_score']:.3f}, final={scene['final_score']:.3f}",
                    f"- Caption: {scene.get('caption', '')}",
                    f"- Transcript: {scene.get('transcript', '')}",
                ]
            )
        )
    return "\\n\\n".join(lines)


def _query(movie_id: str, text: str, image_file: str | None, top_k: int, top_scenes: int):
    if not movie_id.strip():
        return "Provide a movie id first (from ingest)."
    if not text.strip() and not image_file:
        return "Provide text and/or image for querying."

    image = Image.open(image_file).convert("RGB") if image_file else None
    result = query_movie(
        movie_id=movie_id.strip(),
        text=text.strip() or None,
        image=image,
        top_k=top_k,
        top_scenes=top_scenes,
    )
    return _render_scenes(result)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="FilmSpot Demo") as app:
        gr.Markdown("## FilmSpot — Multimodal Temporal Video Retrieval")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Ingest")
                movie_file = gr.File(label="Upload movie", file_types=["video"], type="filepath")
                movie_id_input = gr.Textbox(label="Movie ID (optional)", placeholder="auto-generated if empty")
                fps_input = gr.Number(label="FPS", value=1.0, precision=2)
                semantic_k_input = gr.Number(label="Semantic K", value=5, precision=0)
                ingest_btn = gr.Button("Ingest")
                active_movie_id = gr.Textbox(label="Active Movie ID")
                ingest_status = gr.Markdown()

            with gr.Column():
                gr.Markdown("### Query")
                query_text = gr.Textbox(label="Text Query", lines=3)
                query_image = gr.Image(label="Reference Image (optional)", type="filepath")
                top_k_input = gr.Number(label="Top K", value=12, precision=0)
                top_scenes_input = gr.Number(label="Top Scenes", value=3, precision=0)
                query_btn = gr.Button("Search")
                result_md = gr.Markdown()

        ingest_btn.click(
            fn=_ingest,
            inputs=[movie_file, movie_id_input, fps_input, semantic_k_input],
            outputs=[active_movie_id, ingest_status],
        )

        query_btn.click(
            fn=_query,
            inputs=[active_movie_id, query_text, query_image, top_k_input, top_scenes_input],
            outputs=[result_md],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
