# =====================================================================
# SHOT-WISE RADAR (SPIDER) CHART GENERATION (1, 3, 5, 10 SHOT AXES)
# Publication quality: Font size 12 & bold, titles removed, 300 DPI.
# Special handling: Overlapping lines (e.g. Precision = 1.0) are both made visible.
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Categories/Axes of the spider plot (4 shots)
categories = ["1-shot", "3-shot", "5-shot", "10-shot"]
N = len(categories)

# Compute angles for the 4-axis polar plot
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Stage 1 Dataset Metrics (from your tables)
stage1_data = {
    "Precision": {
        "Triplet Loss": [1.0000, 1.0000, 1.0000, 1.0000],
        "ArcFace Loss": [1.0000, 1.0000, 1.0000, 1.0000]
    },
    "Recall": {
        "Triplet Loss": [0.9813, 0.9810, 0.9806, 0.9796],
        "ArcFace Loss": [0.9907, 0.9905, 0.9903, 0.9898]
    },
    "F1-score": {
        "Triplet Loss": [0.9906, 0.9904, 0.9902, 0.9897],
        "ArcFace Loss": [0.9953, 0.9952, 0.9951, 0.9949]
    }
}

# Stage 2 Dataset Metrics (from your tables)
stage2_data = {
    "Precision": {
        "Triplet Loss": [0.9043, 0.9285, 0.9081, 0.8900],
        "ArcFace Loss": [0.9333, 0.9216, 0.9251, 0.9143]
    },
    "Recall": {
        "Triplet Loss": [0.8877, 0.9354, 0.9151, 0.8947],
        "ArcFace Loss": [0.9284, 0.9155, 0.9165, 0.8815]
    },
    "F1-score": {
        "Triplet Loss": [0.8907, 0.9318, 0.9115, 0.8917],
        "ArcFace Loss": [0.9306, 0.9181, 0.9202, 0.8918]
    }
}

# Stylings designed so that perfectly overlapping lines (e.g. Precision = 1.0)
# are both clearly visible (Triplet is thick/solid, ArcFace is thin/dashed on top).
styles = {
    "Triplet Loss": {
        "color": "#FF6B6B",  # Coral Red
        "marker": "o",
        "markersize": 10,
        "linewidth": 4.5,  # Thicker line
        "linestyle": "solid",  # Solid line style
        "fill_alpha": 0.08,
        "zorder": 2  # Drawn underneath
    },
    "ArcFace Loss": {
        "color": "#4D96FF",  # Soft Blue
        "marker": "s",  # Square marker
        "markersize": 6,  # Smaller marker to fit inside Triplet's circle
        "linewidth": 2.0,  # Thinner line
        "linestyle": "dashed",  # Dashed line style
        "fill_alpha": 0.08,
        "zorder": 3  # Drawn on top
    }
}

# Configuration for Stage plotting ranges
stage_configs = {
    "stage1": {
        "data": stage1_data,
        "yticks": [0.97, 0.98, 0.99, 1.00],
        "yticklabels": ["0.97", "0.98", "0.99", "1.00"],
        "ylim": (0.965, 1.005)
    },
    "stage2": {
        "data": stage2_data,
        "yticks": [0.88, 0.90, 0.92, 0.94],
        "yticklabels": ["0.88", "0.90", "0.92", "0.94"],
        "ylim": (0.87, 0.95)
    }
}

print("Generating spider (radar) plots for Stage 1 and Stage 2...")

# Loop through each stage and metric
for stage_name, config in stage_configs.items():
    for metric, methods in config["data"].items():
        # Setup Figure with Polar Projection
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Orient 1-shot at the top (12 o'clock / 90 degrees) and rotate clockwise
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set tick labels for the axes (1-shot, 3-shot, 5-shot, 10-shot) in bold size 12
        plt.xticks(angles[:-1], categories, fontsize=12, weight='bold')

        # Position the radial value labels at 180 degrees (straight down) to avoid overlaps
        ax.set_rlabel_position(180)

        # Configure radial ticks
        plt.yticks(
            config["yticks"],
            config["yticklabels"],
            color="#555555",
            fontsize=10,
            weight='bold'
        )
        plt.ylim(config["ylim"])

        # Plot each loss model
        for model_name, values in methods.items():
            # Append start value to close the loop
            values_closed = values + [values[0]]

            # Plot the line and marker points
            ax.plot(
                angles,
                values_closed,
                linewidth=styles[model_name]["linewidth"],
                linestyle=styles[model_name]["linestyle"],
                label=model_name,
                color=styles[model_name]["color"],
                marker=styles[model_name]["marker"],
                markersize=styles[model_name]["markersize"],
                zorder=styles[model_name]["zorder"]
            )
            # Fill the interior polygon area
            ax.fill(
                angles,
                values_closed,
                color=styles[model_name]["color"],
                alpha=styles[model_name]["fill_alpha"],
                zorder=styles[model_name]["zorder"]
            )

        # Customize gridlines and spine appearance for premium publication quality
        ax.grid(True, color="#CCCCCC", linestyle="--", linewidth=0.8)
        ax.spines['polar'].set_color('#888888')

        # Position legend centrally at the bottom
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=True,
            facecolor='white',
            edgecolor='#DDDDDD',
            prop={'size': 12, 'weight': 'bold'}
        )

        # Adjust layout and save high-resolution figure
        metric_fname = metric.lower().replace("-", "")
        save_path = f"outputs/{stage_name}_radar_{metric_fname}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" -> Saved: {save_path}")

print("All spider plots generated successfully!")