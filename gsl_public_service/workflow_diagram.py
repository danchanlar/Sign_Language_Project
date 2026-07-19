import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap

steps = [
    ("1. Dataset Access", 
     "Use GSL_continuous.tar.gz and GSL_continuous_files/*.csv from Google Drive."),

    ("2. Read Annotations", 
     "Each CSV row contains: video_key | gloss_sentence."),

    ("3. Dataset Structure", 
     "Each GSL sentence is stored as a folder of .jpg frames, not as one video file."),

    ("4. Dataset Analysis", 
     "Analyze sentence instances, unique gloss sentences, gloss vocabulary, signers and repetitions."),

    ("5. Subset Selection", 
     "Select a small subset, e.g. 20 sentence classes × 10 samples per class."),

    ("6. Match Frame Folders", 
     "Match each video_key with the corresponding frame folder inside the tar.gz file."),

    ("7. Extract Frames", 
     "Extract only the selected frame folders, without extracting the full 37GB dataset."),

    ("8. Sort Frames", 
     "Sort frames in temporal order: frame_0000.jpg, frame_0001.jpg, etc."),

    ("9. Fixed-Length Sequences", 
     "Convert each sentence to 60 frames by sampling or padding with the last frame."),

    ("10. MediaPipe Processing", 
     "Extract hand and body landmarks from each selected frame using MediaPipe Holistic."),

    ("11. Landmark Sequence", 
     "Create input sequences with shape: 60 frames × 258 features."),

    ("12. Model Training", 
     "Train a sequence model such as GRU, LSTM or Transformer for sentence classification."),

    ("13. Evaluation", 
     "Evaluate with accuracy, top-k accuracy, classification report and confusion matrix."),

    ("14. Export Artifacts", 
     "Save model.keras, label_encoder.pkl, normalization_stats.npz and app_config.json."),

    ("15. Streamlit App", 
     "Upload GSL input, predict gloss sentence and show possible natural Greek meaning.")
]

# Figure settings
fig_width = 15
fig_height = 30
box_width = 9.0
box_height = 1.2
vertical_gap = 0.55

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis("off")

x = 1.0
y_start = len(steps) * (box_height + vertical_gap)

ax.set_xlim(0, 11)
ax.set_ylim(0, y_start + 1)

for i, (title, description) in enumerate(steps):
    y = y_start - i * (box_height + vertical_gap)

    # Box
    box = FancyBboxPatch(
        (x, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(box)

    # Text inside box
    wrapped_description = textwrap.fill(description, width=65)
    text = f"{title}\n{wrapped_description}"

    ax.text(
        x + box_width / 2,
        y + box_height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10
    )

    # Arrow
    if i < len(steps) - 1:
        arrow = FancyArrowPatch(
            (x + box_width / 2, y),
            (x + box_width / 2, y - vertical_gap + 0.08),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.4
        )
        ax.add_patch(arrow)

plt.title(
    "Greek Sign Language Public-Service Recognition Workflow",
    fontsize=18,
    pad=25
)

plt.tight_layout()
plt.savefig("gsl_public_service_workflow_clean.png", dpi=200, bbox_inches="tight")
plt.show()