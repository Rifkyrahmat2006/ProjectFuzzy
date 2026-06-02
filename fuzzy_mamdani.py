from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from skfuzzy import control as ctrl


warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")


DATA_PATH = Path(__file__).with_name("student_sleep_patterns.csv")


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(
        df[["Sleep_Duration", "Physical_Activity", "Caffeine_Intake", "Sleep_Quality"]]
    )
    return pd.DataFrame(
        normalized_data,
        columns=["Sleep_Duration", "Physical_Activity", "Caffeine_Intake", "Sleep_Quality"],
    )


@lru_cache(maxsize=1)
def build_fuzzy_system() -> tuple[ctrl.ControlSystem, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent]:
    sleep_duration = ctrl.Antecedent(np.arange(4, 9.1, 0.1), "Sleep_Duration")
    activity = ctrl.Antecedent(np.arange(0, 120.1, 1), "Physical_Activity")
    caffeine = ctrl.Antecedent(np.arange(0, 5.1, 0.1), "Caffeine_Intake")
    sleep_quality = ctrl.Consequent(np.arange(1, 10.1, 0.1), "Sleep_Quality")

    sleep_duration["pendek"] = fuzz.trapmf(sleep_duration.universe, [4, 4, 6.5, 6.5])
    sleep_duration["sedang"] = fuzz.trimf(sleep_duration.universe, [5, 6.5, 8])
    sleep_duration["panjang"] = fuzz.trapmf(sleep_duration.universe, [6.5, 9, 9, 9])

    activity["rendah"] = fuzz.trapmf(activity.universe, [0, 0, 60, 60])
    activity["sedang"] = fuzz.trimf(activity.universe, [30, 60, 90])
    activity["tinggi"] = fuzz.trapmf(activity.universe, [60, 120, 120, 120])

    caffeine["rendah"] = fuzz.trapmf(caffeine.universe, [0, 0, 2.5, 2.5])
    caffeine["sedang"] = fuzz.trimf(caffeine.universe, [1, 2.5, 4])
    caffeine["tinggi"] = fuzz.trapmf(caffeine.universe, [3, 5, 5, 5])

    sleep_quality["buruk"] = fuzz.trapmf(sleep_quality.universe, [1, 1, 5, 5])
    sleep_quality["cukup"] = fuzz.trimf(sleep_quality.universe, [3, 5, 7])
    sleep_quality["baik"] = fuzz.trimf(sleep_quality.universe, [5, 10, 10])
    sleep_quality["sangat_baik"] = fuzz.trimf(sleep_quality.universe, [7, 9, 10])

    rules = [
        ctrl.Rule(sleep_duration["pendek"] & activity["rendah"] & caffeine["tinggi"], sleep_quality["buruk"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["rendah"] & caffeine["sedang"], sleep_quality["buruk"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["rendah"] & caffeine["rendah"], sleep_quality["buruk"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["sedang"] & caffeine["tinggi"], sleep_quality["buruk"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["sedang"] & caffeine["sedang"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["sedang"] & caffeine["rendah"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["tinggi"] & caffeine["tinggi"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["tinggi"] & caffeine["sedang"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["pendek"] & activity["tinggi"] & caffeine["rendah"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["rendah"] & caffeine["tinggi"], sleep_quality["buruk"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["rendah"] & caffeine["sedang"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["rendah"] & caffeine["rendah"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["sedang"] & caffeine["tinggi"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["sedang"] & caffeine["sedang"], sleep_quality["baik"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["sedang"] & caffeine["rendah"], sleep_quality["baik"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["tinggi"] & caffeine["tinggi"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["tinggi"] & caffeine["sedang"], sleep_quality["baik"]),
        ctrl.Rule(sleep_duration["sedang"] & activity["tinggi"] & caffeine["rendah"], sleep_quality["sangat_baik"]),
        ctrl.Rule(sleep_duration["panjang"] & activity["rendah"] & caffeine["tinggi"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["panjang"] & activity["rendah"] & caffeine["sedang"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["panjang"] & activity["rendah"] & caffeine["rendah"], sleep_quality["baik"]),
        ctrl.Rule(sleep_duration["panjang"] & activity["sedang"] & caffeine["tinggi"], sleep_quality["cukup"]),
        ctrl.Rule(sleep_duration["panjang"] & activity["sedang"] & caffeine["sedang"], sleep_quality["baik"]),
        ctrl.Rule(sleep_duration["panjang"] & activity["sedang"] & caffeine["rendah"], sleep_quality["sangat_baik"]),
    ]

    return ctrl.ControlSystem(rules), sleep_duration, activity, caffeine, sleep_quality


def get_rule_base() -> list[dict[str, str]]:
    return [
        {"durasi": "pendek", "aktivitas": "rendah", "kafein": "tinggi", "output": "buruk"},
        {"durasi": "pendek", "aktivitas": "rendah", "kafein": "sedang", "output": "buruk"},
        {"durasi": "pendek", "aktivitas": "rendah", "kafein": "rendah", "output": "buruk"},
        {"durasi": "pendek", "aktivitas": "sedang", "kafein": "tinggi", "output": "buruk"},
        {"durasi": "pendek", "aktivitas": "sedang", "kafein": "sedang", "output": "cukup"},
        {"durasi": "pendek", "aktivitas": "sedang", "kafein": "rendah", "output": "cukup"},
        {"durasi": "pendek", "aktivitas": "tinggi", "kafein": "tinggi", "output": "cukup"},
        {"durasi": "pendek", "aktivitas": "tinggi", "kafein": "sedang", "output": "cukup"},
        {"durasi": "pendek", "aktivitas": "tinggi", "kafein": "rendah", "output": "cukup"},
        {"durasi": "sedang", "aktivitas": "rendah", "kafein": "tinggi", "output": "buruk"},
        {"durasi": "sedang", "aktivitas": "rendah", "kafein": "sedang", "output": "cukup"},
        {"durasi": "sedang", "aktivitas": "rendah", "kafein": "rendah", "output": "cukup"},
        {"durasi": "sedang", "aktivitas": "sedang", "kafein": "tinggi", "output": "cukup"},
        {"durasi": "sedang", "aktivitas": "sedang", "kafein": "sedang", "output": "baik"},
        {"durasi": "sedang", "aktivitas": "sedang", "kafein": "rendah", "output": "baik"},
        {"durasi": "sedang", "aktivitas": "tinggi", "kafein": "tinggi", "output": "cukup"},
        {"durasi": "sedang", "aktivitas": "tinggi", "kafein": "sedang", "output": "baik"},
        {"durasi": "sedang", "aktivitas": "tinggi", "kafein": "rendah", "output": "sangat_baik"},
        {"durasi": "panjang", "aktivitas": "rendah", "kafein": "tinggi", "output": "cukup"},
        {"durasi": "panjang", "aktivitas": "rendah", "kafein": "sedang", "output": "cukup"},
        {"durasi": "panjang", "aktivitas": "rendah", "kafein": "rendah", "output": "baik"},
        {"durasi": "panjang", "aktivitas": "sedang", "kafein": "tinggi", "output": "cukup"},
        {"durasi": "panjang", "aktivitas": "sedang", "kafein": "sedang", "output": "baik"},
        {"durasi": "panjang", "aktivitas": "sedang", "kafein": "rendah", "output": "sangat_baik"},
    ]


def _antecedent_memberships(variable: ctrl.Antecedent, crisp_value: float) -> dict[str, float]:
    return {
        label: float(fuzz.interp_membership(variable.universe, membership.mf, crisp_value))
        for label, membership in variable.terms.items()
    }


def _plot_antecedent(ax: plt.Axes, variable: ctrl.Antecedent, crisp_value: float, title: str, color: str) -> None:
    for label, membership in variable.terms.items():
        ax.plot(variable.universe, membership.mf, linewidth=2, label=label)

    ax.axvline(crisp_value, color=color, linestyle="--", linewidth=2, label="input")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="upper left")


def _aggregate_output(
    sleep_duration: ctrl.Antecedent,
    activity: ctrl.Antecedent,
    caffeine: ctrl.Antecedent,
    sleep_quality: ctrl.Consequent,
    durasi_tidur: float,
    aktivitas_fisik: float,
    kafein: float,
) -> tuple[np.ndarray, float]:
    duration_memberships = _antecedent_memberships(sleep_duration, durasi_tidur)
    activity_memberships = _antecedent_memberships(activity, aktivitas_fisik)
    caffeine_memberships = _antecedent_memberships(caffeine, kafein)

    aggregated = np.zeros_like(sleep_quality.universe, dtype=float)
    for rule in get_rule_base():
        rule_strength = min(
            duration_memberships[rule["durasi"]],
            activity_memberships[rule["aktivitas"]],
            caffeine_memberships[rule["kafein"]],
        )
        output_membership = sleep_quality.terms[rule["output"]].mf
        aggregated = np.maximum(aggregated, np.fmin(rule_strength, output_membership))

    centroid = float(fuzz.defuzz(sleep_quality.universe, aggregated, "centroid"))
    return aggregated, centroid


def evaluate_sleep_quality(durasi_tidur: float, aktivitas_fisik: float, kafein: float) -> tuple[float, ctrl.ControlSystemSimulation]:
    sleep_ctrl, *_ = build_fuzzy_system()
    simulation = ctrl.ControlSystemSimulation(sleep_ctrl)
    simulation.input["Sleep_Duration"] = durasi_tidur
    simulation.input["Physical_Activity"] = aktivitas_fisik
    simulation.input["Caffeine_Intake"] = kafein
    simulation.compute()
    return float(simulation.output["Sleep_Quality"]), simulation


def build_visualization_figure(
    durasi_tidur: float,
    aktivitas_fisik: float,
    kafein: float,
    simulation: ctrl.ControlSystemSimulation | None = None,
) -> plt.Figure:
    sleep_ctrl, sleep_duration, activity, caffeine, sleep_quality = build_fuzzy_system()
    if simulation is None:
        simulation = ctrl.ControlSystemSimulation(sleep_ctrl)
        simulation.input["Sleep_Duration"] = durasi_tidur
        simulation.input["Physical_Activity"] = aktivitas_fisik
        simulation.input["Caffeine_Intake"] = kafein
        simulation.compute()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    _plot_antecedent(axes[0, 0], sleep_duration, durasi_tidur, "Durasi Tidur", "tab:red")
    _plot_antecedent(axes[0, 1], activity, aktivitas_fisik, "Aktivitas Fisik", "tab:green")
    _plot_antecedent(axes[1, 0], caffeine, kafein, "Kafein", "tab:blue")

    aggregated, centroid = _aggregate_output(
        sleep_duration,
        activity,
        caffeine,
        sleep_quality,
        durasi_tidur,
        aktivitas_fisik,
        kafein,
    )
    axes[1, 1].plot(sleep_quality.universe, sleep_quality["buruk"].mf, linestyle="--", alpha=0.3, label="buruk")
    axes[1, 1].plot(sleep_quality.universe, sleep_quality["cukup"].mf, linestyle="--", alpha=0.3, label="cukup")
    axes[1, 1].plot(sleep_quality.universe, sleep_quality["baik"].mf, linestyle="--", alpha=0.3, label="baik")
    axes[1, 1].plot(sleep_quality.universe, sleep_quality["sangat_baik"].mf, linestyle="--", alpha=0.3, label="sangat_baik")
    axes[1, 1].fill_between(sleep_quality.universe, 0, aggregated, color="tab:purple", alpha=0.35, label="hasil agregasi")
    axes[1, 1].axvline(centroid, color="black", linestyle="--", linewidth=2, label=f"centroid {centroid:.2f}")
    axes[1, 1].set_title("Kualitas Tidur")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.2)
    axes[1, 1].legend(fontsize=7, loc="upper left")

    fig.suptitle("Visualisasi Lengkap Fuzzy Mamdani", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def run_cli_demo() -> None:
    df = load_dataset()
    df_norm = normalize_dataset(df)

    print("Data setelah normalisasi:")
    print(df_norm.head())
    print("\n=== 24 Percobaan Prediksi Sleep Quality ===")

    for i in range(min(24, len(df))):
        sample = df.iloc[i]
        hasil, _ = evaluate_sleep_quality(
            float(sample["Sleep_Duration"]),
            float(sample["Physical_Activity"]),
            float(sample["Caffeine_Intake"]),
        )
        print(
            f"Percobaan {i + 1}: Durasi={sample['Sleep_Duration']} jam, Aktivitas={sample['Physical_Activity']}, "
            f"Kafein={sample['Caffeine_Intake']} → Prediksi Kualitas Tidur = {hasil:.2f}"
        )

    fig, axes = plt.subplots(6, 4, figsize=(24, 18))
    axes = axes.flatten()

    sleep_ctrl, _, _, _, sleep_quality = build_fuzzy_system()

    for i in range(min(24, len(df))):
        sample = df.iloc[i]
        _, simulation = evaluate_sleep_quality(
            float(sample["Sleep_Duration"]),
            float(sample["Physical_Activity"]),
            float(sample["Caffeine_Intake"]),
        )
        sleep_quality.view(sim=simulation, ax=axes[i])
        axes[i].set_title(f"Percobaan {i + 1}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cli_demo()
