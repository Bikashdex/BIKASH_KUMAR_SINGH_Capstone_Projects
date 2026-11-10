"""
Drift Monitor v4 - Evidently AI (any version) + PSI verdict engine.
Compares REFERENCE (training) vs CURRENT (live log) data and
auto-triggers retrain.py when drift crosses the threshold.
"""
import pandas as pd
import numpy as np
import subprocess, sys, os
import importlib

# ---------- version-agnostic import ----------
def _find(mods, cls):
    for m in mods:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, cls):
                return getattr(mod, cls)
        except Exception:
            continue
    return None

# ---------- object -> dict (tries every known API) ----------
def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    for meth in ("as_dict", "to_dict"):
        if hasattr(obj, meth):
            try:
                return getattr(obj, meth)()
            except Exception:
                pass
    if hasattr(obj, "items"):
        try:
            return dict(obj.items())
        except Exception:
            pass
    return None

# ---------- deep search for a key inside nested dicts/objects ----------
def _search(obj, key, depth=0):
    if depth > 4:
        return None
    d = _to_dict(obj)
    if isinstance(d, dict):
        if key in d:
            return d
        for v in d.values():
            r = _search(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(d, (list, tuple)):
        for v in d:
            r = _search(v, key, depth + 1)
            if r is not None:
                return r
    return None

# ---------- PSI: the banking-industry standard drift metric ----------
def psi_numeric(ref, cur, bins=10):
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    ref_pct = np.clip(np.histogram(ref, bins=q)[0] / len(ref), 1e-4, None)
    cur_pct = np.clip(np.histogram(cur, bins=q)[0] / len(cur), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def psi_categorical(ref, cur):
    ref_pct = ref.value_counts(normalize=True)
    cur_pct = cur.value_counts(normalize=True)
    cats = sorted(set(ref_pct.index) | set(cur_pct.index), key=str)
    r = np.clip([ref_pct.get(c, 0.0) for c in cats], 1e-4, None)
    c = np.clip([cur_pct.get(c, 0.0) for c in cats], 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))

# ---------- 1. Load REFERENCE and CURRENT data ----------
ref = pd.read_csv('train_transaction.csv', nrows=5000)
ref = ref.rename(columns={'TransactionAmt': 'amount', 'ProductCD': 'card_type'})
ref = ref[['amount', 'card_type']].fillna({'amount': 0.0, 'card_type': 'W'})

cur = pd.read_csv('predictions_log.csv')[['amount', 'card_type']].dropna()
print(f"Reference rows: {len(ref)} | Current rows: {len(cur)}")
if len(cur) < 30:
    print("Not enough live data yet. Run kafka_consumer_v2.py first.")
    sys.exit()

# ---------- 2. Run Evidently (requirement) + try HTML save ----------
Report = _find(["evidently.report", "evidently.core.report", "evidently"], "Report")
DataDriftPreset = _find(["evidently.metric_preset", "evidently.core.metric_preset",
                         "evidently.presets", "evidently"], "DataDriftPreset")
os.makedirs("reports", exist_ok=True)
html_path = os.path.join("reports", "drift_report.html")

ev_share = None
if Report and DataDriftPreset:
    try:
        report = Report(metrics=[DataDriftPreset()])
        run_out = report.run(reference_data=ref, current_data=cur)
        info = (_search(run_out, "number_of_drifted_columns")
                or _search(report, "number_of_drifted_columns"))
        if info:
            drifted = info.get("number_of_drifted_columns", 0)
            total = info.get("number_of_columns", 0)
            ev_share = drifted / total if total else 0.0
            print(f"Evidently: drifted {drifted}/{total} columns")
            for col, cinfo in (info.get("columns") or {}).items():
                if isinstance(cinfo, dict):
                    print(f"  - {col}: {'DRIFTED' if cinfo.get('drift_detected') else 'stable'}")
        saved = False
        for obj in (run_out, report):
            if saved or obj is None:
                continue
            for meth in ("save_html", "save", "to_html"):
                if hasattr(obj, meth):
                    try:
                        if meth == "to_html":
                            with open(html_path, "w", encoding="utf-8") as f:
                                f.write(obj.to_html())
                        else:
                            getattr(obj, meth)(html_path)
                        saved = True
                        break
                    except Exception:
                        continue
        print("HTML report saved to reports/drift_report.html" if saved
              else "HTML save unsupported in this evidently version (PSI verdict below still valid).")
    except Exception as e:
        print("Evidently run failed:", e)
else:
    print("Evidently not importable - using PSI only.")

# ---------- 3. PSI verdict (always real numbers) ----------
print("\n--- PSI Drift Check (banking industry standard) ---")
psi_results = {
    'amount': psi_numeric(ref['amount'].values, cur['amount'].values),
    'card_type': psi_categorical(ref['card_type'], cur['card_type']),
}
drifted_cols = [c for c, v in psi_results.items() if v > 0.2]
for col, val in psi_results.items():
    status = "DRIFTED" if val > 0.2 else ("watch" if val > 0.1 else "stable")
    print(f"  - {col}: PSI = {val:.4f} -> {status}")
psi_share = len(drifted_cols) / len(psi_results)

# ---------- 4. Final verdict + automated retrain loop ----------
share = ev_share if ev_share is not None else psi_share
print(f"\nOverall drift share: {share:.0%}")
THRESHOLD = 0.5
if share >= THRESHOLD:
    print("🚨 RETRAIN TRIGGER: drift crossed threshold!")
    if os.path.exists("retrain.py"):
        subprocess.run([sys.executable, "retrain.py"])
    else:
        print("retrain.py not found - create it to enable auto-retraining.")
else:
    print(f"✅ No retrain needed (drift {share:.0%} < threshold {THRESHOLD:.0%}).")