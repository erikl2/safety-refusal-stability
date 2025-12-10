"""
Safety Stability Explorer

Interactive demo showing how safety decisions flip across random seeds and temperatures.
Visualizes the core finding: 32% of harmful prompts produce inconsistent safety decisions.
"""

import gradio as gr
import json
from pathlib import Path

# Load data
DATA_PATH = Path(__file__).parent / "data" / "prompts.json"


def load_data():
    """Load the prompts data."""
    with open(DATA_PATH) as f:
        return json.load(f)


# Global data
DATA = load_data()
TEMPS = DATA["metadata"]["temps"]
SEEDS = DATA["metadata"]["seeds"]


def get_dropdown_choices():
    """Build dropdown choices sorted by SSI ascending (most unstable first)."""
    choices = []
    for prompt in DATA["prompts"]:
        ssi = prompt["ssi"]
        text = prompt["text"][:50] + "..." if len(prompt["text"]) > 50 else prompt["text"]
        display = f"[SSI: {ssi:.2f}] {text}"
        choices.append((display, prompt["id"]))
    return choices


def get_prompt_by_id(prompt_id):
    """Get prompt data by ID."""
    for prompt in DATA["prompts"]:
        if prompt["id"] == prompt_id:
            return prompt
    return None


def get_ssi_color(ssi):
    """Get color class based on SSI value."""
    if ssi >= 0.8:
        return "#22c55e"  # green
    elif ssi >= 0.6:
        return "#eab308"  # yellow
    else:
        return "#ef4444"  # red


def get_label_color(label):
    """Get color for label."""
    colors = {
        "REFUSE": "#22c55e",   # green
        "PARTIAL": "#eab308",  # yellow
        "COMPLY": "#ef4444",   # red
    }
    return colors.get(label, "#6b7280")


def create_grid_html(prompt_data):
    """Generate HTML for the response grid."""
    if not prompt_data:
        return "<p>Select a prompt to view the response grid.</p>"

    # Build lookup for responses
    response_map = {}
    for r in prompt_data["responses"]:
        key = (r["temp"], r["seed"])
        response_map[key] = r

    html = """
    <style>
        .grid-container {
            display: flex;
            flex-direction: column;
            gap: 4px;
            font-family: system-ui, -apple-system, sans-serif;
        }
        .grid-row {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .grid-header {
            width: 70px;
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            color: #9ca3af;
        }
        .grid-cell {
            width: 70px;
            height: 40px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 500;
            color: white;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.1s;
            border: 2px solid transparent;
        }
        .grid-cell:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .temp-label {
            width: 70px;
            text-align: right;
            padding-right: 8px;
            font-size: 12px;
            color: #9ca3af;
            font-weight: 500;
        }
    </style>
    <div class="grid-container">
    """

    # Header row with seeds
    html += '<div class="grid-row"><div class="temp-label"></div>'
    for seed in SEEDS:
        html += f'<div class="grid-header">Seed {seed}</div>'
    html += '</div>'

    # Data rows
    for temp in TEMPS:
        html += f'<div class="grid-row"><div class="temp-label">T={temp}</div>'
        for seed in SEEDS:
            r = response_map.get((temp, seed))
            if r:
                label = r["label"]
                color = get_label_color(label)
                short_label = label[0]  # R, P, or C
                html += f'<div class="grid-cell" style="background-color: {color};" data-temp="{temp}" data-seed="{seed}">{short_label}</div>'
            else:
                html += '<div class="grid-cell" style="background-color: #374151;">?</div>'
        html += '</div>'

    html += '</div>'

    # Legend
    html += """
    <div style="margin-top: 16px; display: flex; gap: 16px; font-size: 12px; color: #9ca3af;">
        <span><span style="display: inline-block; width: 12px; height: 12px; background: #22c55e; border-radius: 2px; margin-right: 4px;"></span>REFUSE</span>
        <span><span style="display: inline-block; width: 12px; height: 12px; background: #eab308; border-radius: 2px; margin-right: 4px;"></span>PARTIAL</span>
        <span><span style="display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 2px; margin-right: 4px;"></span>COMPLY</span>
    </div>
    """

    return html


def format_response_display(prompt_data, temp, seed):
    """Format response text for display."""
    if not prompt_data:
        return "Select a prompt first."

    for r in prompt_data["responses"]:
        if r["temp"] == temp and r["seed"] == seed:
            return f"**Label:** {r['label']}\n\n**Response:**\n{r['text']}"

    return "Response not found."


def on_prompt_select(prompt_id):
    """Handle prompt selection."""
    prompt_data = get_prompt_by_id(prompt_id)
    if not prompt_data:
        return (
            "",  # category
            "",  # ssi_display
            "<p>Prompt not found.</p>",  # grid
            "",  # prompt_text
            "Select cells to compare",  # left panel
            "Select cells to compare",  # right panel
        )

    # SSI display with color
    ssi = prompt_data["ssi"]
    ssi_color = get_ssi_color(ssi)
    ssi_html = f'<span style="font-size: 48px; font-weight: bold; color: {ssi_color};">{ssi:.2f}</span>'

    # Category badge
    cat = prompt_data["category"].replace("_", " ").title()
    cat_html = f'<span style="background: #374151; padding: 4px 12px; border-radius: 9999px; font-size: 14px; color: #d1d5db;">{cat}</span>'

    # Grid
    grid_html = create_grid_html(prompt_data)

    # Full prompt text
    prompt_text = prompt_data["text"]

    # Default comparison panels - pick first two different responses
    responses = prompt_data["responses"]
    left_text = format_response_display(prompt_data, TEMPS[0], SEEDS[0])

    # Find a different response for comparison
    first_label = responses[0]["label"] if responses else None
    right_r = next((r for r in responses if r["label"] != first_label), responses[1] if len(responses) > 1 else responses[0])
    right_text = format_response_display(prompt_data, right_r["temp"], right_r["seed"]) if right_r else "Select a cell"

    return (
        cat_html,
        ssi_html,
        grid_html,
        prompt_text,
        left_text,
        right_text,
    )


def get_response_for_cell(prompt_id, temp_idx, seed_idx):
    """Get response for a specific cell."""
    prompt_data = get_prompt_by_id(prompt_id)
    if not prompt_data:
        return "Select a prompt first."

    temp = TEMPS[temp_idx]
    seed = SEEDS[seed_idx]
    return format_response_display(prompt_data, temp, seed)


# Build the app
def build_app():
    """Build and return the Gradio app."""

    custom_css = """
        .main-title { font-size: 28px !important; font-weight: 700 !important; margin-bottom: 4px !important; }
        .subtitle { color: #9ca3af !important; font-size: 16px !important; margin-bottom: 24px !important; }
        .prompt-text { background: #1e293b !important; padding: 16px !important; border-radius: 8px !important; font-family: monospace !important; }
        .comparison-box { min-height: 200px !important; }
        .ssi-container { text-align: center; }
    """

    with gr.Blocks(title="Safety Stability Explorer", css=custom_css) as app:

        # Header
        gr.Markdown("# Safety Stability Explorer", elem_classes=["main-title"])
        gr.Markdown("**32% of harmful prompts flip between refuse and comply across random seeds**", elem_classes=["subtitle"])

        # Prompt selector
        dropdown = gr.Dropdown(
            choices=get_dropdown_choices(),
            label="Select Prompt",
            value=DATA["prompts"][0]["id"] if DATA["prompts"] else None,
            interactive=True,
        )

        # Info row
        with gr.Row():
            with gr.Column(scale=1):
                category_display = gr.HTML(label="Category")
            with gr.Column(scale=1):
                ssi_display = gr.HTML(label="SSI Score", elem_classes=["ssi-container"])

        # Full prompt text
        prompt_text = gr.Textbox(
            label="Full Prompt",
            lines=2,
            interactive=False,
            elem_classes=["prompt-text"],
        )

        # Response grid
        gr.Markdown("### Response Grid")
        gr.Markdown("*Each cell shows the model's decision for that temperature × seed combination*")
        grid_display = gr.HTML()

        # Cell selectors for comparison
        gr.Markdown("### Compare Responses")
        gr.Markdown("*Select temperature and seed to view the full response*")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response A**")
                temp_a = gr.Dropdown(
                    choices=[(f"T={t}", i) for i, t in enumerate(TEMPS)],
                    label="Temperature",
                    value=0,
                )
                seed_a = gr.Dropdown(
                    choices=[(f"Seed {s}", i) for i, s in enumerate(SEEDS)],
                    label="Seed",
                    value=0,
                )
                response_a = gr.Markdown(elem_classes=["comparison-box"])

            with gr.Column():
                gr.Markdown("**Response B**")
                temp_b = gr.Dropdown(
                    choices=[(f"T={t}", i) for i, t in enumerate(TEMPS)],
                    label="Temperature",
                    value=2,  # T=0.7
                )
                seed_b = gr.Dropdown(
                    choices=[(f"Seed {s}", i) for i, s in enumerate(SEEDS)],
                    label="Seed",
                    value=1,  # Seed 43
                )
                response_b = gr.Markdown(elem_classes=["comparison-box"])

        # State for current prompt
        current_prompt_id = gr.State(value=DATA["prompts"][0]["id"] if DATA["prompts"] else None)

        # Event handlers
        def update_display(prompt_id):
            result = on_prompt_select(prompt_id)
            return (prompt_id,) + result

        def update_response_a(prompt_id, temp_idx, seed_idx):
            return get_response_for_cell(prompt_id, temp_idx, seed_idx)

        def update_response_b(prompt_id, temp_idx, seed_idx):
            return get_response_for_cell(prompt_id, temp_idx, seed_idx)

        dropdown.change(
            fn=update_display,
            inputs=[dropdown],
            outputs=[current_prompt_id, category_display, ssi_display, grid_display, prompt_text, response_a, response_b],
        )

        # Update response A when selectors change
        temp_a.change(
            fn=update_response_a,
            inputs=[current_prompt_id, temp_a, seed_a],
            outputs=[response_a],
        )
        seed_a.change(
            fn=update_response_a,
            inputs=[current_prompt_id, temp_a, seed_a],
            outputs=[response_a],
        )

        # Update response B when selectors change
        temp_b.change(
            fn=update_response_b,
            inputs=[current_prompt_id, temp_b, seed_b],
            outputs=[response_b],
        )
        seed_b.change(
            fn=update_response_b,
            inputs=[current_prompt_id, temp_b, seed_b],
            outputs=[response_b],
        )

        # Initial load
        app.load(
            fn=update_display,
            inputs=[dropdown],
            outputs=[current_prompt_id, category_display, ssi_display, grid_display, prompt_text, response_a, response_b],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
