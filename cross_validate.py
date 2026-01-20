import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN threads (optional but safer on HPC)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import logging
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from config import CFG
from data_loader import VideoDataset
from model import cusModel, PreModel
from trainer import Trainer


def build_dataset_from_arrays(features, targets):
    # features: numpy array, targets: 1D array-like
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.shuffle(CFG.batch_size * 4).batch(CFG.batch_size).cache().prefetch(1)

    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 1
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    ds = ds.with_options(options)
    return ds


def main(n_splits=5, save_dir="cv_outputs", subset_frac=1.0):
    os.makedirs(save_dir, exist_ok=True)

    # load all data once
    dataset = VideoDataset()
    dataset.load_files()
    features = dataset.extract_features()
    targets = np.array(dataset.targets)

    total_samples = len(targets)
    print(f"Total samples: {total_samples}")

    # optionally sample a stratified subset of the data first
    if subset_frac < 1.0:
        if subset_frac <= 0.0:
            raise ValueError("--subset must be > 0 and <= 1.0")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=subset_frac, random_state=CFG.random_state)
        _, subset_idx = next(sss.split(features, targets))
        features = features[subset_idx]
        targets = targets[subset_idx]
        print(f"Using subset: {len(targets)} samples ({subset_frac*100:.1f}% of total)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.random_state)

    records = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, targets), start=1):
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")
        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]

        train_ds = build_dataset_from_arrays(x_train, y_train)
        val_ds = build_dataset_from_arrays(x_val, y_val)

        # build a fresh model per fold
        if CFG.model_select == "custom_model":
            model = cusModel.build()
        elif CFG.model_select == "pre_model":
            model = PreModel.build()
            # ensure model is built
            try:
                dummy_input = tf.random.normal((1, CFG.n_frames, CFG.output_size[0], CFG.output_size[1], 3))
                _ = model(dummy_input)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown model type: {CFG.model_select}")

        ckpt_path = os.path.join(save_dir, f"model_fold_{fold_idx}.h5")
        trainer = Trainer(model, train_ds, val_ds, checkpoint_path=ckpt_path, plot_dir=save_dir, run_name=f"fold{fold_idx}")

        history = trainer.train()
        val_loss, val_acc = trainer.evaluate()

        # save history arrays
        hist_file = os.path.join(save_dir, f"fold_{fold_idx}_history.npz")
        np.savez_compressed(hist_file, **{k: np.array(v) for k, v in history.history.items()})

        records.append({
            "fold": fold_idx,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "ckpt_path": ckpt_path,
            "history_file": hist_file,
        })

    # save cv results summary
    results_df = pd.DataFrame.from_records(records)
    results_csv = os.path.join(save_dir, "cv_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved CV summary to: {results_csv}")

    # pick best fold
    best_row = results_df.loc[results_df["val_acc"].idxmax()]
    best_fold = int(best_row["fold"])
    best_ckpt = best_row["ckpt_path"]

    print(f"Best fold: {best_fold} (ckpt: {best_ckpt})")

    # Retrain on full dataset using the same model architecture; load best weights if available
    print("\nRetraining on full dataset using best fold weights (if available)...")
    # build full dataset
    full_ds = build_dataset_from_arrays(features, targets)

    if CFG.model_select == "custom_model":
        final_model = cusModel.build()
    else:
        final_model = PreModel.build()
        try:
            dummy_input = tf.random.normal((1, CFG.n_frames, CFG.output_size[0], CFG.output_size[1], 3))
            _ = final_model(dummy_input)
        except Exception:
            pass

    # load best fold weights if file exists
    if os.path.exists(best_ckpt):
        try:
            final_model.load_weights(best_ckpt)
            print(f"Loaded weights from best fold: {best_ckpt}")
        except Exception as e:
            print(f"Unable to load best fold weights: {e}")

    full_ckpt = os.path.join(save_dir, "model_full.h5")
    final_trainer = Trainer(final_model, full_ds, full_ds, checkpoint_path=full_ckpt, plot_dir=save_dir, run_name="full")
    full_history = final_trainer.train()
    # evaluate on full data (not ideal, but provides final metrics)
    loss, acc = final_trainer.evaluate()

    final_hist_file = os.path.join(save_dir, "full_history.npz")
    np.savez_compressed(final_hist_file, **{k: np.array(v) for k, v in full_history.history.items()})

    final_summary = {
        "best_fold": int(best_fold),
        "best_ckpt": best_ckpt,
        "final_ckpt": full_ckpt,
        "final_val_loss": float(loss),
        "final_val_acc": float(acc),
    }

    pd.Series(final_summary).to_json(os.path.join(save_dir, "final_summary.json"))
    print(f"Saved final summary and models to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-fold cross validation and retrain on full data")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross validation (e.g. 5)")
    parser.add_argument("--out", type=str, default="cv_outputs", help="Output directory for CV artifacts")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to sample before CV (e.g. 0.1 = 10%%). Default 1.0 uses all data.")
    args = parser.parse_args()

    main(n_splits=args.folds, save_dir=args.out, subset_frac=args.subset)
