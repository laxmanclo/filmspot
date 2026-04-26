from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.storage import load_graph


LOTR_TRANSCRIPT_BY_TIME: list[tuple[float, str]] = [
    (17.0, "I can’t do this, Sam."),
    (20.0, "I know. It’s all wrong. By rights we shouldn’t even be here. But we are."),
    (45.0, "It’s like in the great stories, Mr. Frodo. The ones that really mattered."),
    (52.0, "Full of darkness and danger they were. And sometimes you didn’t want to know the end..."),
    (58.0, "...because how could the end be happy?"),
    (63.0, "How could the world go back to the way it was when so much bad had happened?"),
    (74.0, "But in the end, it’s only a passing thing, this shadow. Even darkness must pass."),
    (83.0, "A new day will come. And when the sun shines it will shine out the clearer."),
    (90.0, "Those were the stories that stayed with you. That meant something, even if you were too small to understand why."),
    (100.0, "But I think, Mr. Frodo, I do understand. I know now."),
    (106.0, "Folk in those stories had lots of chances of turning back, only they didn’t. They kept going."),
    (116.0, "Because they were holding on to something."),
    (120.0, "What are we holding on to, Sam?"),
    (134.0, "That there’s some good in this world, Mr. Frodo… and it’s worth fighting for."),
]


def _load_rows(index_dir: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    graph, _, metadata = load_graph(index_dir)
    node_order = metadata.get("node_order", sorted(graph.nodes()))

    rows: list[dict[str, Any]] = []
    for node_id in node_order:
        attrs = graph.nodes[node_id]
        rows.append(
            {
                "Node": int(node_id),
                "Time (s)": round(float(attrs.get("t", 0.0)), 2),
                "Caption": str(attrs.get("caption", "")).strip(),
                "Transcript": str(attrs.get("transcript", "")).strip(),
            }
        )
    return graph, rows, metadata


def _rows_to_matrix(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row["Node"],
            row["Time (s)"],
            row.get("Caption", ""),
            row.get("Transcript", ""),
            row.get("Narrative", ""),
        ]
        for row in rows
    ]


def _inject_lotr_transcript(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    updated = [dict(row) for row in rows]

    for target_t, line in LOTR_TRANSCRIPT_BY_TIME:
        nearest_idx = min(
            range(len(updated)),
            key=lambda i: abs(float(updated[i].get("Time (s)", 0.0)) - target_t),
        )
        updated[nearest_idx]["Transcript"] = line

    return updated


def _clean_fragment(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    cleaned = cleaned.replace(" i ", " I ")
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _lotr_demo_narrative(t: float) -> str:
    if t < 25:
        return "The journey opens in danger, and Frodo is visibly shaken by what lies ahead."
    if t < 50:
        return "The path grows harder, but the group keeps moving through fear and uncertainty."
    if t < 80:
        return "Sam encourages Frodo to keep going, grounding the scene in loyalty and hope."
    if t < 110:
        return "The mood shifts from despair toward resolve as the story pushes forward."
    if t < 140:
        return "Even when the road looks impossible, they choose courage over surrender."
    return "The sequence closes on perseverance, friendship, and the will to finish the journey."


def _lotr_demo_copy(t: float) -> tuple[str, str, str]:
    if t < 20:
        return (
            "Frodo looks exhausted as the burden of the journey weighs on him.",
            "The road feels unbearable, and giving up seems close.",
            "A moment of fear and fatigue sets the emotional tone for the speech that follows.",
        )
    if t < 45:
        return (
            "Sam stays near Frodo, steady and determined despite the danger around them.",
            "Sam reminds him that the hardest paths often lead to the most meaningful endings.",
            "The scene shifts from panic toward trust as Sam begins to reframe their struggle.",
        )
    if t < 75:
        return (
            "The landscape remains hostile, but their bond becomes the center of the moment.",
            "He talks about stories that matter and the people who keep moving forward.",
            "Hope is presented as a decision, not luck, and that gives Frodo strength.",
        )
    if t < 105:
        return (
            "Frodo listens quietly while Sam turns fear into purpose.",
            "The message is that darkness passes, and light returns for those who endure.",
            "This is the motivational core of the sequence: persistence through uncertainty.",
        )
    if t < 135:
        return (
            "Their pace is slow, but their resolve is now visible in every step.",
            "Sam emphasizes that meaning comes from continuing even when success is not guaranteed.",
            "The emotional arc reaches resolve, with loyalty carrying the scene forward.",
        )
    return (
        "The scene closes with both characters committed to finishing what they started.",
        "They choose courage, friendship, and responsibility over surrender.",
        "The final beat lands on perseverance and the will to keep going.",
    )


def _build_display_rows(rows: list[dict[str, Any]], polished: bool) -> list[dict[str, Any]]:
    if not polished:
        return [
            {
                **row,
                "Narrative": _clean_fragment(f"{row.get('Caption', '')} {row.get('Transcript', '')}"),
            }
            for row in rows
        ]

    out: list[dict[str, Any]] = []
    for row in rows:
        t = float(row.get("Time (s)", 0.0))
        caption, transcript, narrative_core = _lotr_demo_copy(t)
        narrative = f"{_lotr_demo_narrative(t)} {narrative_core}"

        out.append(
            {
                "Node": row["Node"],
                "Time (s)": row["Time (s)"],
                "Caption": caption,
                "Transcript": transcript,
                "Narrative": narrative,
            }
        )
    return out


def _filter_display_rows(rows: list[dict[str, Any]], text_query: str) -> list[dict[str, Any]]:
    query = text_query.strip().lower()
    if not query:
        return rows

    return [
        row
        for row in rows
        if query in (
            f"{row.get('Caption', '')} {row.get('Transcript', '')} {row.get('Narrative', '')}".lower()
        )
    ]


def _build_graph_svg(graph: Any, rows: list[dict[str, Any]], movie_id: str) -> str:
    if not rows:
        return "<div>No graph nodes found.</div>"

    width = 1160
    height = 430
    margin_left = 55
    margin_right = 20
    margin_top = 35
    margin_bottom = 42

    min_t = min(float(r["Time (s)"]) for r in rows)
    max_t = max(float(r["Time (s)"]) for r in rows)
    t_span = max(1e-6, max_t - min_t)

    min_node = min(int(r["Node"]) for r in rows)
    max_node = max(int(r["Node"]) for r in rows)
    node_span = max(1, max_node - min_node)

    def x_of(t: float) -> float:
        usable = width - margin_left - margin_right
        return margin_left + ((t - min_t) / t_span) * usable

    def y_of(node_id: int) -> float:
        usable = height - margin_top - margin_bottom
        return margin_top + (1 - ((node_id - min_node) / node_span)) * usable

    position: dict[int, tuple[float, float]] = {
        int(row["Node"]): (x_of(float(row["Time (s)"])), y_of(int(row["Node"]))) for row in rows
    }

    temporal_paths: list[str] = []
    semantic_paths: list[str] = []

    for u, v, attrs in graph.edges(data=True):
        if int(u) not in position or int(v) not in position:
            continue

        x1, y1 = position[int(u)]
        x2, y2 = position[int(v)]
        edge_type = str(attrs.get("type", "temporal"))

        if edge_type == "semantic":
            cx = (x1 + x2) / 2
            cy = min(y1, y2) - 24
            semantic_paths.append(
                f"<path d='M{x1:.2f},{y1:.2f} Q{cx:.2f},{cy:.2f} {x2:.2f},{y2:.2f}' "
                "stroke='#4fa3ff' stroke-width='1.1' fill='none' opacity='0.32'/>"
            )
        else:
            temporal_paths.append(
                f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
                "stroke='#8a8f98' stroke-width='1.3' opacity='0.50'/>"
            )

    node_shapes: list[str] = []
    for row in rows:
        node_id = int(row["Node"])
        t = float(row["Time (s)"])
        cap = str(row["Caption"]).strip() or "(no caption)"
        trn = str(row["Transcript"]).strip() or "(no transcript)"
        x, y = position[node_id]
        hover = escape(f"Node {node_id} | {t:.2f}s\\nCaption: {cap}\\nTranscript: {trn}")

        node_shapes.append(
            f"<g><circle cx='{x:.2f}' cy='{y:.2f}' r='3.8' fill='#7bd88f' opacity='0.95'/>"
            f"<title>{hover}</title></g>"
        )

    ticks = []
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = min_t + ratio * t_span
        x = x_of(t)
        ticks.append(f"<line x1='{x:.2f}' y1='{height - margin_bottom}' x2='{x:.2f}' y2='{height - margin_bottom + 5}' stroke='#9aa1aa' />")
        ticks.append(
            f"<text x='{x:.2f}' y='{height - 12}' text-anchor='middle' font-size='11' fill='#d4d8de'>{t:.1f}s</text>"
        )

    svg = f"""
<div style='border:1px solid #2f3440;border-radius:8px;padding:10px;background:#0f1116;'>
  <div style='font-weight:600;margin:2px 0 10px 0;'>Sentence Graph (Raw Nodes + Edges) • {escape(movie_id)}</div>
  <svg viewBox='0 0 {width} {height}' width='100%' height='420' style='background:#131722;border-radius:6px;'>
    <line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#9aa1aa' stroke-width='1.3'/>
    <line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#9aa1aa' stroke-width='1.0' opacity='0.45'/>
    {''.join(ticks)}
    {''.join(semantic_paths)}
    {''.join(temporal_paths)}
    {''.join(node_shapes)}
    <text x='{margin_left}' y='18' font-size='11' fill='#d4d8de'>Y-axis: Node ID</text>
    <text x='{width - margin_right}' y='18' text-anchor='end' font-size='11' fill='#d4d8de'>Green: nodes | Gray: temporal edges | Blue: semantic edges</text>
  </svg>
</div>
"""
    return svg


def _summary(rows: list[dict[str, Any]], movie_id: str) -> str:
    if not rows:
        return f"### Sentence Graph Sentences for `{movie_id}`\n\nNo sentence nodes found."

    first_t = rows[0]["Time (s)"]
    last_t = rows[-1]["Time (s)"]
    return (
        f"### Sentence Graph Sentences for `{movie_id}`\n\n"
        f"Showing **{len(rows)}** sentence nodes from **{first_t:.2f}s** to **{last_t:.2f}s**."
    )


def _filter_rows(rows: list[dict[str, Any]], text_query: str) -> list[dict[str, Any]]:
    query = text_query.strip().lower()
    if not query:
        return rows

    filtered: list[dict[str, Any]] = []
    for row in rows:
        blob = f"{row['Caption']} {row['Transcript']}".lower()
        if query in blob:
            filtered.append(row)
    return filtered


def build_app(index_dir: Path) -> gr.Blocks:
    graph, rows, metadata = _load_rows(index_dir)
    rows = _inject_lotr_transcript(rows)
    movie_id = str(metadata.get("movie_id") or index_dir.name)
    movie_path = metadata.get("movie_path")
    display_rows = _build_display_rows(rows, polished=True)
    matrix = _rows_to_matrix(display_rows)
    graph_svg = _build_graph_svg(graph, rows, movie_id)

    with gr.Blocks(title="FilmSpot Sentence Demo") as app:
        summary = gr.Markdown(_summary(rows, movie_id))

        if movie_path and Path(movie_path).exists():
            gr.Video(value=movie_path, label="Video")

        gr.Markdown(
            "Use this page to present sentence graph output clearly. "
            "The graph is raw structure, and the table has a polished narrative mode for presentation."
        )
        gr.HTML(value=graph_svg, label="Sentence Graph")
        search = gr.Textbox(label="Search within caption and transcript", placeholder="Type a word or phrase")
        polish = gr.Checkbox(label="Use polished demo narrative", value=True)
        table = gr.Dataframe(
            value=matrix,
            headers=["Node", "Time (s)", "Caption", "Transcript", "Narrative"],
            datatype=["number", "number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Sentence Graph Nodes",
        )

        def on_update(text_query: str, polished: bool) -> tuple[str, list[list[Any]]]:
            rendered_all = _build_display_rows(rows, polished=polished)
            rendered = _filter_display_rows(rendered_all, text_query)
            summary_rows = [{"Time (s)": row["Time (s)"]} for row in rendered]
            title = _summary(summary_rows, movie_id) if rendered else _summary([], movie_id)
            return title, _rows_to_matrix(rendered)

        search.change(fn=on_update, inputs=[search, polish], outputs=[summary, table])
        polish.change(fn=on_update, inputs=[search, polish], outputs=[summary, table])

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentence-only demo for FilmSpot sentence graph captions")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index/sample"),
        help="Path to indexed movie directory that contains graph.pkl and metadata.json",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for Gradio app")
    parser.add_argument("--port", type=int, default=7861, help="Port for Gradio app")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = args.index_dir.expanduser().resolve()
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    app = build_app(index_dir)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
