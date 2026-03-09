import argparse
import glob
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluation import Evaluator


CHO_CLASS_NAMES = [
    "cystic_plate",
    "calot_triangle",
    "cystic_artery",
    "cystic_duct",
    "gallbladder",
    "tool",
]


def load_model(model_config_path: str, checkpoint_path: str, dataloaders_cfg=None):
    model_cfg = OmegaConf.load(model_config_path)
    if dataloaders_cfg is not None:
        container = OmegaConf.create()
        container.dataloaders = dataloaders_cfg
        container.model = model_cfg
        cfg = container.model
    else:
        cfg = model_cfg
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    try:
        out_features = state_dict["global_memory.prior_encoder.fc.weight"].shape[0]
        d_latent = cfg.global_memory.prior_encoder.d_latent
        denom = 2 * d_latent + 1
        if out_features % denom == 0:
            inferred_clusters = out_features // denom
            if cfg.global_memory.prior_encoder.num_clusters != inferred_clusters:
                print(
                    f"Adjusting num_clusters: {cfg.global_memory.prior_encoder.num_clusters} -> {inferred_clusters}"
                )
                cfg.global_memory.prior_encoder.num_clusters = inferred_clusters
    except Exception:
        pass

    model = instantiate(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()
    return model


def plot_radar_multi(class_names, model_scores, title, save_path):
    num_classes = len(class_names)
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)

    for name, scores in model_scores.items():
        values = list(scores) + [scores[0]]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.08)

    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/zheyaogao/Experiments/ESD_seg/checkpoints",
        help="Directory containing CHO model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="CHO_*.pth",
        help="Glob pattern for CHO checkpoints inside checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths (overrides checkpoint-dir/glob).",
    )
    parser.add_argument(
        "--model-config",
        default="/home/zheyaogao/Experiments/ESD_seg/cfgs/model.yaml",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--output-dice",
        default="/home/zheyaogao/Experiments/ESD_seg/vis_results/CHO_dice_radar.png",
        help="Path to save Dice radar chart.",
    )
    parser.add_argument(
        "--output-fscore",
        default="/home/zheyaogao/Experiments/ESD_seg/vis_results/CHO_fscore_radar.png",
        help="Path to save F-score radar chart.",
    )
    args = parser.parse_args()

    # Ensure local modules can be imported when running from elsewhere.
    sys.path.append(os.getcwd())

    try:
        GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="cfgs")
        cfg = compose(config_name="experiments")
    except Exception as exc:
        raise RuntimeError(f"Hydra initialization error: {exc}") from exc

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    cfg.dataloaders.batch_size = args.batch_size
    cfg.dataloaders.num_workers = args.num_workers

    dataloaders = instantiate(cfg.dataloaders)
    test_loader = dataloaders.test_loader

    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    else:
        pattern = os.path.join(args.checkpoint_dir, args.checkpoint_glob)
        checkpoint_paths = sorted([p for p in glob.glob(pattern)])

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found for CHO evaluation.")

    evaluator = Evaluator(num_cls=dataloaders.num_classes, smooth=1e-5)
    per_class_dice_by_model = {}
    per_class_f1_by_model = {}

    class_names = CHO_CLASS_NAMES

    for ckpt_path in checkpoint_paths:
        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"Evaluating {model_name}...")
        model = load_model(args.model_config, ckpt_path, cfg.dataloaders)
        res, _ = evaluator(model, test_loader, validation=False, return_details=True)
        per_class_dice = res["per_class_dice"]
        per_class_f1 = res["per_class_f1"]

        if len(per_class_dice) != len(CHO_CLASS_NAMES):
            class_names = [f"class_{i+1}" for i in range(len(per_class_dice))]

        per_class_dice_by_model[model_name] = per_class_dice
        per_class_f1_by_model[model_name] = per_class_f1

        print("  Per-class Dice:")
        for name, score in zip(class_names, per_class_dice):
            print(f"    {name}: {score:.4f}")
        print("  Per-class F-score:")
        for name, score in zip(class_names, per_class_f1):
            print(f"    {name}: {score:.4f}")

    plot_radar_multi(
        class_names,
        per_class_dice_by_model,
        "CHO per-class Dice (all checkpoints)",
        args.output_dice,
    )
    print(f"Dice radar chart saved to {args.output_dice}")

    plot_radar_multi(
        class_names,
        per_class_f1_by_model,
        "CHO per-class F-score (all checkpoints)",
        args.output_fscore,
    )
    print(f"F-score radar chart saved to {args.output_fscore}")


if __name__ == "__main__":
    main()
