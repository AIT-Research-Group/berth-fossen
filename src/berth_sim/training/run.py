"""Run a reproducible synthetic forecasting experiment end to end."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .data import prepare_dataset, load_split
from .model import fit_scaler
from .train import train_model, configure_device
from .evaluation import evaluate


def plots(output, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ("lstm_point", "energy_lstm", "gan_lstm")
    import csv
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, name in zip(axes, names):
        with (output/name/"history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        axis.plot([int(row["epoch"]) for row in rows], [float(row["validation_score"]) for row in rows])
        axis.set(title=name, xlabel="Epoch", ylabel="Validation MSE" if name == "lstm_point" else "Validation energy score")
        axis.grid(alpha=.2)
    figure.savefig(output/"evaluation"/"training_curves.png", dpi=160)
    plt.close(figure)
    with np.load(output/"evaluation"/"examples.npz") as examples:
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for axis, variable, label in zip(axes, (2, 3), ("Gust speed (m/s)", "Reflected Hm0 (m)")):
            history = examples["history"][0, :, variable]
            axis.plot(np.arange(-len(history)+1, 1), history, color="black", label="Observed history")
            future = np.arange(1, examples["target"].shape[1]+1)
            axis.plot(future, examples["target"][0, :, variable], color="black", linestyle="--", label="Observed future")
            for name, color in (("energy_lstm", "tab:blue"), ("gan_lstm", "tab:orange")):
                samples = examples[name][0, :, :, variable]
                lower, upper = np.quantile(samples, [.05, .95], axis=0)
                axis.plot(future, samples.mean(axis=0), label=name, color=color)
                axis.fill_between(future, lower, upper, alpha=.15, color=color)
            axis.axvline(0, color="gray", linewidth=.8)
            axis.set(xlabel="Minutes relative to forecast issuance", ylabel=label)
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        figure.suptitle("Held-out synthetic scenario; 90% marginal intervals")
        figure.savefig(output/"evaluation"/"forecast_example.png", dpi=160)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, variable in zip(axes, ("gust_mps", "reflected_hs_m")):
        rows = [row for row in metrics if row["horizon_min"] == 30 and row["variable"] == variable]
        axis.bar([row["model"] for row in rows], [row["rmse"] for row in rows])
        axis.tick_params(axis="x", rotation=25)
        axis.set(title=f"30-minute {variable}", ylabel="RMSE")
    figure.savefig(output/"evaluation"/"rmse_comparison.png", dpi=160)
    plt.close(figure)


def report(output, config, manifest, summaries, metrics, warning, joint):
    rows = ["# Kết quả huấn luyện trên dữ liệu mô phỏng", "",
        "Đây là thí nghiệm ban đầu trên dữ liệu tổng hợp từ bộ mô phỏng môi trường dùng trong khung Fossen. Không phải kiểm chứng bằng dữ liệu cảng thực.", "",
        f"Lịch sử {config['history_steps']} phút; đầu ra {config['horizon_steps']} phút; seed {config['seed']}; thiết bị {config['device']}.", "",
        "| Tập | Kịch bản độc lập | Cửa sổ |", "|---|---:|---:|"]
    for name, split in manifest["splits"].items():
        rows.append(f"| {name} | {split['scenarios']} | {split['windows']} |")
    rows += ["", "Chuẩn hóa chỉ fit trên các dòng đặc trưng duy nhất của tập train. Chọn checkpoint bằng validation; fit hiệu chỉnh xác suất bằng calibration; test chỉ dùng đánh giá cuối cùng.", "",
        "| Mô hình | Epoch chạy | Điểm validation tốt nhất |", "|---|---:|---:|"]
    for name, summary in summaries.items():
        rows.append(f"| {name} | {summary['epochs_run']} | {summary['best_validation_score']:.5f} |")
    rows += ["", "Validation của LSTM điểm là MSE; hai mô hình sinh dùng energy score. Không so trực tiếp hai loại điểm này.", "",
        "| Mô hình | Biến | Chân trời (phút) | RMSE | CRPS | Coverage 90% |", "|---|---|---:|---:|---:|---:|"]
    for row in metrics:
        if row["variable"] not in ("gust_mps", "reflected_hs_m"):
            continue
        crps = f"{row['crps']:.4f}" if row["crps"] != "" else "—"
        coverage = f"{100*row['coverage_90']:.1f}%" if row["coverage_90"] != "" else "—"
        rows.append(f"| {row['model']} | {row['variable']} | {row['horizon_min']} | {row['rmse']:.4f} | {crps} | {coverage} |")
    rows += ["", "| Mô hình sinh | Energy score toàn chuỗi, chuẩn hóa |", "|---|---:|"]
    for row in joint:
        rows.append(f"| {row['model']} | {row['normalized_trajectory_energy_score']:.5f} |")
    rows += ["", "Các ngưỡng dưới đây chỉ là ngưỡng thử nghiệm, không phải giới hạn vận hành cảng:",
        f"gió giật > {config['demo_thresholds']['gust_mps']} m/s; Hm0 phản xạ > {config['demo_thresholds']['reflected_hs_m']} m.", "",
        "| Mô hình | Sự kiện trong 30 phút | Brier thô | Brier sau hiệu chỉnh | Mốc dự báo dương tính | Kịch bản có nhãn dương |", "|---|---|---:|---:|---:|---:|"]
    for row in warning:
        if row["horizon_min"] == 30:
            rows.append(f"| {row['model']} | {row['event']} | {row['brier_raw']:.4f} | {row['brier_calibrated']:.4f} | {row['positive_origins']} | {row['positive_scenarios']} |")
    rows += ["", "## Giới hạn diễn giải", "",
        "- Mỗi kịch bản dài 4 giờ; gió và sóng thay đổi theo một họ xung thời tiết quy định trước. Chia tập độc lập không loại bỏ thiên lệch do cùng họ mô phỏng.",
        "- Lần chạy dùng một seed huấn luyện. Cần nhiều seed, nhiều họ kịch bản và dữ liệu đo để kết luận về tính vượt trội.",
        "- Tắt mô phỏng tàu khi tạo dữ liệu AI vì biến đích chỉ là môi trường. Dữ liệu này không phải lực hay chuyển động tàu đã hiệu chỉnh.",
        "- Đích Hm0 dùng cửa sổ lùi 10 phút: tại mốc +5 phút còn chồng lấn lịch sử, tại +10/+30 phút không còn phần đo đã biết. Persistence dùng cùng định nghĩa đích.",
        "- Coverage là biên theo biến/chân trời. Hiệu chỉnh xác suất sự kiện không hiệu chỉnh khoảng dự báo liên tục.",
        "- Số mốc dự báo dương tính không phải số sự kiện độc lập. CSV có khoảng tin cậy bootstrap theo toàn bộ kịch bản; chỉ 8 kịch bản test nên độ chắc chắn còn hạn chế.",
        "- Chưa tính cảnh báo theo đợt, thời gian báo trước hoặc báo động giả theo giờ. Không suy diễn Brier thành chứng nhận hiệu quả vận hành.", "",
        "## Các tệp kết quả", "",
        "- `evaluation/forecast_metrics.csv`: sai số, CRPS, coverage và khoảng tin cậy RMSE.",
        "- `evaluation/warning_metrics.csv`: Brier và số mốc/kịch bản dương tính.",
        "- `evaluation/reliability.csv`: xác suất dự báo, tần suất quan sát và số mẫu mỗi bin.",
        "- `*/best.pt`: trọng số tốt nhất, cấu hình, chuẩn hóa và epoch.",
        "- `*/history.csv`: lịch sử loss và validation.",
        "- `source/`: bản mã nguồn dùng cho lần chạy.", "",
        "![Đường huấn luyện](evaluation/training_curves.png)", "",
        "![So sánh RMSE](evaluation/rmse_comparison.png)", "",
        "![Ví dụ dự báo](evaluation/forecast_example.png)"]
    (output/"report.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_v1"))
    parser.add_argument("--dataset", type=Path, help="reuse (or create) a shared dataset directory")
    parser.add_argument("--seed", type=int, help="training seed; the dataset keeps its data_seed")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.seed is not None:
        config.setdefault("data_seed", config["seed"])
        config["seed"] = args.seed
    if config["horizon_steps"] != 30 or config["history_steps"] < 1:
        raise ValueError("Evaluation currently requires horizon=30 and a positive history")
    if config["ensemble_train"] < 2 or config["ensemble_test"] < 2:
        raise ValueError("Distributional training/evaluation require >=2 ensemble samples")
    if args.output.joinpath("experiment.json").exists():
        raise FileExistsError("Experiment exists; choose a new output directory")
    configure_device(config)
    started = time.monotonic()
    dataset = args.dataset or args.output/"dataset"
    manifest = prepare_dataset(config, dataset)
    experiment = {"config": config, "dataset": str(dataset), "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "matplotlib")},
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if config["device"].startswith("cuda") else None,
                  "synthetic_only": True}
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source, args.output/"source"/"berth_sim", ignore=shutil.ignore_patterns("__pycache__"))
    experiment["source_sha256"] = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.py")}
    (args.output/"experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    splits = {name: load_split(dataset, name) for name in ("train", "validation", "calibration", "test")}
    scaler = fit_scaler(splits["train"]["unique_features"])
    (args.output/"scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    summaries = {}
    for name in ("lstm_point", "energy_lstm", "gan_lstm"):
        summaries[name] = train_model(name, config, scaler, splits["train"], splits["validation"], args.output)
    metrics, warning, joint = evaluate(config, scaler, splits["calibration"], splits["test"], args.output)
    plots(args.output, metrics)
    report(args.output, config, manifest, summaries, metrics, warning, joint)
    print(json.dumps({"report": str(args.output/"report.md"), "elapsed_s": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
