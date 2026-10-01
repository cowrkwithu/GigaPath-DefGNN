"""Stage 7 — Visualization (Phase 9)."""

from src.visualization.figures import (
    PALETTE,
    generate_fig01_pipeline,
    generate_fig02_graph_variants,
    generate_fig03_baseline_bars,
    generate_fig04_graph_radar,
    generate_fig05_confusion_matrix,
    generate_fig06_roc,
    generate_fig07_attention_overlays,
    generate_fig08_tsne,
    generate_fig09_transfer,
    generate_figS1_training_curves,
)
from src.visualization.tables import (
    generate_table01_dataset,
    generate_table02_baseline,
    generate_table03_graphs,
    generate_table04_backbones,
    generate_table05_fusion,
    generate_table06_transfer,
    generate_table07_per_class,
    generate_table08_cost,
)

FIGURE_REGISTRY = {
    "fig01_pipeline": generate_fig01_pipeline,
    "fig02_graph_variants": generate_fig02_graph_variants,
    "fig03_baseline_bars": generate_fig03_baseline_bars,
    "fig04_graph_radar": generate_fig04_graph_radar,
    "fig05_confusion": generate_fig05_confusion_matrix,
    "fig06_roc": generate_fig06_roc,
    "fig07_attention": generate_fig07_attention_overlays,
    "fig08_tsne": generate_fig08_tsne,
    "fig09_transfer": generate_fig09_transfer,
    "figS1_curves": generate_figS1_training_curves,
}

DEFERRED_FIGURES = ("fig01_pipeline", "fig02_graph_variants", "fig07_attention")

TABLE_REGISTRY = {
    "table01_dataset": generate_table01_dataset,
    "table02_baseline": generate_table02_baseline,
    "table03_graphs": generate_table03_graphs,
    "table04_backbones": generate_table04_backbones,
    "table05_fusion": generate_table05_fusion,
    "table06_transfer": generate_table06_transfer,
    "table07_per_class": generate_table07_per_class,
    "table08_cost": generate_table08_cost,
}

__all__ = [
    "DEFERRED_FIGURES",
    "FIGURE_REGISTRY",
    "PALETTE",
    "TABLE_REGISTRY",
    "generate_fig01_pipeline",
    "generate_fig02_graph_variants",
    "generate_fig03_baseline_bars",
    "generate_fig04_graph_radar",
    "generate_fig05_confusion_matrix",
    "generate_fig06_roc",
    "generate_fig07_attention_overlays",
    "generate_fig08_tsne",
    "generate_fig09_transfer",
    "generate_figS1_training_curves",
    "generate_table01_dataset",
    "generate_table02_baseline",
    "generate_table03_graphs",
    "generate_table04_backbones",
    "generate_table05_fusion",
    "generate_table06_transfer",
    "generate_table07_per_class",
    "generate_table08_cost",
]
