"""Genere monitoring/drift_report.html avec Evidently 0.4.25."""
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REF = Path("monitoring/reference.csv")
CUR = Path("monitoring/predictions_log.csv")
OUT = Path("monitoring/drift_report.html")

COLS = ["confidence", "brightness", "contrast", "mean_r", "mean_g", "mean_b",
        "width", "height", "prediction"]

def main():
    if not REF.exists():
        print("ERREUR: reference.csv introuvable.")
        print("Lancez d'abord : python -m src.monitoring.build_reference")
        return

    if not CUR.exists():
        print("ERREUR: predictions_log.csv introuvable.")
        print("Lancez l'API et faites au moins 30 predictions.")
        return

    ref = pd.read_csv(REF)
    cur = pd.read_csv(CUR)

    if len(cur) < 30:
        print(f"ERREUR: seulement {len(cur)} predictions. Il en faut 30 minimum.")
        return

    cols = [c for c in COLS if c in ref.columns and c in cur.columns]
    ref = ref[cols]
    cur = cur[cols]

    print(f"Reference : {len(ref)} lignes")
    print(f"Courant   : {len(cur)} lignes")
    print("Generation du rapport...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(OUT))

    print("=" * 55)
    print(f"  Rapport genere : {OUT}")
    print("=" * 55)

if __name__ == "__main__":
    main()
