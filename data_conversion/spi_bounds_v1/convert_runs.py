"""
Konvertierung pilot_doe_v0: exportierte Rohserien -> Detektion/Labels -> SPI-HDF5.

Eingabeformat (pro Run ein Verzeichnis <runs_dir>/<run_id>/):
  meta.json:
    {
      "run_id": "doe_r001", "block": 1,
      "v_slide_mm_s": 4.0, "f_n_mag_N": 7.0,
      "r_gt_nom_mm": 9.25,
      "purpose": "training",            # nur "training" wird konvertiert
      "technical_valid": true,           # false => zensiert, wird übersprungen
      "operator_slip": false,
      "point_start": [px,py,pz,qw,qx,qy,qz]   # sliding_frame, Meter, w-first!
      # optional: "x_r_start_m": absolute x_R bei slide_started (sonst pos[0,0])
    }
  series.npz:
    t      [N]    native Zeitstempel in s, t[0] = slide_started
    pos    [N,3]  x_R, y_R, z_R in m (sliding_frame)
    quat   [N,4]  w, x, y, z
    wrench [N,6]  Fx,Fy,Fz,Mx,My,Mz (interner UR-F/T, ins sliding_frame
                  transformiert und debiast; ATI geht NICHT in den Tensor)

Ausgabe:
  <out_dir>/full/train/pilot_all.h5            (--layout full: alle gültigen Runs)
  <out_dir>/lobo/fold<k>/{train,heldout}/*.h5  (--layout lobo: Leave-one-block-out)
  <out_dir>/labels.csv                         (Labels + Sensitivitätsmetadaten)

Regeln exakt nach pilot_doe_v0.md:
  Raster dt=32 ms, T=640 standardmäßig (explizit konfigurierbar). Bei Trigger wird das erste Rastersample bei/nach t_det
  zum Endsample (p_EOS=1). Ohne Trigger: letztes Rastersample bei/vor dem
  15-mm-Cap, p_EOS=1 und p_succ=0. Padding wiederholt das letzte physische
  Sample, p_EOS bleibt 1. success = trigger_before_cap AND |r_det - r_GT,run|
  <= 1,5 mm AND kein operatorbestätigter Slip. Zensierte Läufe (technischer
  Abbruch) werden ausgeschlossen, nicht als Failure gelabelt.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from detector import PilotDetectorConfig, run_pilot_detector
from hdf_io import save_data
from sliding_prior import PilotSlidingPrior
from sequence_config import SequenceConfig, DT, DEFAULT_HORIZON_SAMPLES

MM = 1e-3
SLIDE_DIST_M = 15 * MM
SUCCESS_TOL_M = 1.5 * MM
SENS_TOLS_MM = (1.0, 2.0)
ANTI_ALIAS_CUTOFF_HZ = 12.0
ANTI_ALIAS_ORDER = 4


def interp_series(t_src, y_src, t_grid):
    t_ref = float(t_src[0])
    t_src_rel = t_src - t_ref
    t_grid_rel = t_grid - t_ref
    if y_src.ndim == 1:
        return np.interp(t_grid_rel, t_src_rel, y_src)
    return np.stack(
        [np.interp(t_grid_rel, t_src_rel, y_src[:, j]) for j in range(y_src.shape[1])],
        axis=1,
    )


def normalize_quat_continuous(quat):
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Quaternion mit Norm nahe null")
    q = quat / norms
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def slerp_quaternion_wxyz(t_src, quat_wxyz, t_grid):
    quat_wxyz = normalize_quat_continuous(quat_wxyz)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rotations = Slerp(t_src - t_src[0], Rotation.from_quat(quat_xyzw))(
        t_grid - t_src[0]
    )
    return normalize_quat_continuous(rotations.as_quat()[:, [3, 0, 1, 2]])


def anti_alias_wrench(t_src, wrench, t_grid):
    """Uniformize, zero-phase low-pass, then sample on the 32-ms grid."""
    native_dt = float(np.median(np.diff(t_src)))
    if native_dt <= 0:
        raise ValueError("Ungültige native Wrench-Zeitbasis")
    native_rate_hz = 1.0 / native_dt
    if native_rate_hz <= 2.5 * ANTI_ALIAS_CUTOFF_HZ:
        raise ValueError(
            f"Native Wrench-Rate {native_rate_hz:.2f} Hz zu klein für Anti-Alias-Filter"
        )
    t_uniform = np.arange(t_src[0], t_src[-1] + 0.5 * native_dt, native_dt)
    wrench_uniform = interp_series(t_src, wrench, t_uniform)
    sos = butter(
        ANTI_ALIAS_ORDER,
        ANTI_ALIAS_CUTOFF_HZ,
        btype="lowpass",
        fs=native_rate_hz,
        output="sos",
    )
    wrench_filtered = sosfiltfilt(sos, wrench_uniform, axis=0)
    return interp_series(t_uniform, wrench_filtered, t_grid)


def validate_native_series(run_id, t, pos, quat, wrench):
    n = len(t)
    if t.shape != (n,) or pos.shape != (n, 3) or quat.shape != (n, 4) or wrench.shape != (n, 6):
        raise ValueError(
            f"{run_id}: ungültige Shapes t={t.shape}, pos={pos.shape}, "
            f"quat={quat.shape}, wrench={wrench.shape}"
        )
    if n < 100:
        raise ValueError(f"{run_id}: nur {n} native Samples")
    if not all(np.all(np.isfinite(values)) for values in (t, pos, quat, wrench)):
        raise ValueError(f"{run_id}: NaN/Inf in nativer Serie")
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{run_id}: Zeitstempel nicht streng monoton")
    if abs(float(t[0])) > 1e-9:
        raise ValueError(f"{run_id}: t[0]={t[0]} statt slide_started=0")


def convert_run(run_dir: str, det_cfg: PilotDetectorConfig, sequence: SequenceConfig = SequenceConfig()):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("purpose", "training") != "training":
        return None, f"übersprungen (purpose={meta.get('purpose')})"
    if not meta.get("technical_valid", False):
        return None, "zensiert (technical_valid=false) — ausgeschlossen, KEIN Failure-Label"

    npz = np.load(os.path.join(run_dir, "series.npz"), allow_pickle=False)
    t = np.asarray(npz["t"], float)
    pos = np.asarray(npz["pos"], float)
    quat = normalize_quat_continuous(np.asarray(npz["quat"], float).copy())
    wrench = np.asarray(npz["wrench"], float)
    validate_native_series(meta["run_id"], t, pos, quat, wrench)

    x_start = float(meta.get("x_r_start_m", pos[0, 0]))
    d = pos[:, 0] - x_start
    if d[-1] < 14.5 * MM or np.min(np.diff(d)) < -0.2 * MM:
        raise ValueError(
            f"{meta['run_id']}: unplausible +x_R-Fahrt, d_end={d[-1] / MM:.3f} mm"
        )
    r_gt_run = meta.get("r_gt_nom_mm", 9.25) * MM - (x_start - pos[0, 0]) \
        if "x_r_start_m" in meta else meta.get("r_gt_nom_mm", 9.25) * MM
    # Standardfall: Serie beginnt bei slide_started -> r_GT,run = r_GT,nom - 0.
    # Wenn der Startversatz separat gemessen wurde, r_gt_run_mm direkt in meta setzen:
    if "r_gt_run_mm" in meta:
        r_gt_run = meta["r_gt_run_mm"] * MM

    # --- Detektion auf nativen Stempeln (vor Resampling) ---
    det = run_pilot_detector(t, d, pos[:, 2], det_cfg)

    # --- Cap-Zeit (15 mm) ---
    reach = np.nonzero(d >= SLIDE_DIST_M)[0]
    t_cap = t[reach[0]] if len(reach) else t[-1]

    # --- Endsample auf dem 32-ms-Raster ---
    if det.triggered and det.t_det <= t_cap:
        end_idx = int(np.ceil(det.t_det / DT - 1e-9))     # erstes Sample bei/nach t_det
        triggered = True
    else:
        end_idx = int(np.floor(t_cap / DT + 1e-9))        # letztes Sample bei/vor Cap
        triggered = False
    end_idx = max(1, end_idx)
    if end_idx >= sequence.horizon_samples:
        event = "native detector event" if triggered else "15-mm cap without detection"
        raise ValueError(
            f"{meta['run_id']}: {event} requires EOS index {end_idx} "
            f"({end_idx * DT:.3f} s), beyond horizon {sequence.horizon_samples} "
            f"(last sample {sequence.last_time_s:.3f} s). Increase --horizon-samples "
            "for the entire dataset; no EOS or label has been truncated."
        )
    t_grid = np.arange(end_idx + 1) * DT
    if t_grid[-1] > t[-1] + 1e-9:
        raise ValueError(
            f"{meta['run_id']}: first grid sample after native detection is at "
            f"{t_grid[-1]:.3f} s, beyond observed data {t[-1]:.3f} s; "
            "do not extrapolate the physical EOS sample"
        )

    # --- Labels ---
    e_r = (det.r_det_m - r_gt_run) if triggered else np.nan
    slip = bool(meta.get("operator_slip", False))
    success = bool(triggered and abs(e_r) <= SUCCESS_TOL_M and not slip)
    sens = {f"within_{tol:.1f}mm": bool(triggered and abs(e_r) <= tol * MM and not slip)
            for tol in SENS_TOLS_MM}

    # --- Resampling + 15D-Tensor ---
    pos_g = interp_series(t, pos, t_grid)
    quat_g = slerp_quaternion_wxyz(t, quat, t_grid)
    wr_g = anti_alias_wrench(t, wrench, t_grid)
    n = len(t_grid)
    eos = np.zeros((n, 1)); eos[-1, 0] = 1.0
    succ = np.full((n, 1), 1.0 if success else 0.0)
    traj = np.concatenate([eos, succ, pos_g, quat_g, wr_g], axis=1).astype(np.float32)
    # Padding: letztes physisches Sample wiederholen, p_EOS bleibt 1
    if n < sequence.horizon_samples:
        pad = np.repeat(traj[-1:], sequence.horizon_samples - n, axis=0)
        pad[:, 0] = 1.0
        traj = np.concatenate([traj, pad], axis=0)
    if traj.shape != (sequence.horizon_samples, 15) or not np.all(np.isfinite(traj)):
        raise ValueError(f"{meta['run_id']}: ungültiger 15D-Tensor {traj.shape}")

    x_vec = np.array([meta["v_slide_mm_s"] * MM, meta["f_n_mag_N"]], dtype=np.float32)
    s_in = np.array(meta["point_start"], dtype=np.float32)
    label = dict(run_id=meta["run_id"], block=meta.get("block"),
                 planned_order_index=meta.get("planned_order_index"),
                 v_slide_mm_s=meta["v_slide_mm_s"], f_n_mag_N=meta["f_n_mag_N"],
                 triggered=triggered,
                 t_det_s=round(det.t_det, 5) if triggered else "",
                 r_det_mm=round(det.r_det_m / MM, 4) if triggered else "",
                 r_gt_run_mm=round(r_gt_run / MM, 4),
                 e_r_mm=round(e_r / MM, 4) if triggered else "",
                 operator_slip=slip, success=success,
                 end_idx=end_idx, duration_s=round(end_idx * DT, 3), **sens)
    return dict(x=x_vec, s_in=s_in, traj=traj, label=label,
                r_gt_run=r_gt_run, meta=meta, sequence=sequence), None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_h5(entries, out_path, fz_sign):
    configs = {entry["sequence"] for entry in entries}
    if len(configs) != 1:
        raise ValueError("Cannot write an empty or mixed-horizon HDF5 dataset")
    sequence = configs.pop()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise FileExistsError(f"Ausgabedatei existiert bereits: {out_path}")
    xs, sins, sims, reals = [], [], [], []
    for e in entries:
        prior = PilotSlidingPrior(dt=DT, T=sequence.horizon_samples, slide_dist_m=SLIDE_DIST_M,
                                  r_gt_run_m=e["r_gt_run"], fz_sign=fz_sign)
        with torch.no_grad():
            sim = prior.simulate(torch.from_numpy(e["x"]), torch.from_numpy(e["s_in"]), sequence.horizon_samples)[0]
        xs.append(torch.from_numpy(e["x"]))
        sins.append(torch.from_numpy(e["s_in"]))
        sims.append(sim.float())
        reals.append(torch.from_numpy(e["traj"]))
    save_data(out_path, torch.stack(xs), torch.stack(sins),
              sim=torch.stack(sims), real=torch.stack(reals))
    membership_path = f"{out_path}.runs.json"
    with open(membership_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "h5_file": os.path.basename(out_path),
                "run_ids_in_tensor_order": [e["meta"]["run_id"] for e in entries],
                "sequence": sequence.as_dict(),
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")
    print(f"  {out_path}: {len(entries)} Runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layout", choices=["full", "lobo", "both"], default="both")
    ap.add_argument("--fz-sign", type=float, choices=[-1.0, 1.0], required=True,
                    help="Vorzeichen des Prior-Fz relativ zu F_N_mag (an reale Daten anpassen!)")
    ap.add_argument("--r-det-at", choices=["det", "cross"], default="det")
    ap.add_argument("--horizon-samples", type=int, default=DEFAULT_HORIZON_SAMPLES,
                    help="Shared padded length; 640 samples cover 20.448 s at 32 ms. Overflow is an error.")
    args = ap.parse_args()
    sequence = SequenceConfig(args.horizon_samples)

    if os.path.exists(args.out_dir):
        sys.exit(f"Ausgabeverzeichnis existiert bereits: {args.out_dir}")

    det_cfg = PilotDetectorConfig(r_det_at=args.r_det_at)
    entries, rows = [], []
    for rid in sorted(os.listdir(args.runs_dir)):
        run_dir = os.path.join(args.runs_dir, rid)
        if not os.path.isdir(run_dir):
            continue
        entry, msg = convert_run(run_dir, det_cfg, sequence)
        if entry is None:
            print(f"  {rid}: {msg}")
            continue
        entries.append(entry)
        rows.append(entry["label"])

    if not entries:
        sys.exit("Keine gültigen Runs gefunden.")
    entries.sort(key=lambda item: int(item["meta"].get("planned_order_index", 10**9)))
    rows = [entry["label"] for entry in entries]
    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_succ = sum(r["success"] for r in rows)
    print(f"\n{len(entries)} gültige Runs | success: {n_succ} | failure: {len(entries) - n_succ}")
    if min(n_succ, len(entries) - n_succ) < 5:
        print("WARNUNG (pilot_doe_v0): eine Success-Klasse hat <5 Läufe — "
              "Success-Klassifikation nicht als validiert interpretieren.")

    if args.layout in ("full", "both"):
        write_h5(entries, os.path.join(args.out_dir, "full", "train", "pilot_all.h5"), args.fz_sign)
    if args.layout in ("lobo", "both"):
        blocks = sorted({e["label"]["block"] for e in entries})
        for b in blocks:
            tr = [e for e in entries if e["label"]["block"] != b]
            ho = [e for e in entries if e["label"]["block"] == b]
            write_h5(tr, os.path.join(args.out_dir, "lobo", f"fold{b}", "train", "train.h5"), args.fz_sign)
            write_h5(ho, os.path.join(args.out_dir, "lobo", f"fold{b}", "heldout", "heldout.h5"), args.fz_sign)
    conversion_manifest = {
        "schema_version": "phase_b_spi_bounds_v1_conversion_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs_dir": str(Path(args.runs_dir).resolve()),
        "converter_path": str(Path(__file__).resolve()),
        "converter_sha256": sha256_file(__file__),
        "fz_sign": args.fz_sign,
        "detector": {
            "baseline_max_disp_m": det_cfg.baseline_max_disp_m,
            "min_detect_disp_m": det_cfg.min_detect_disp_m,
            "ema_tau_s": det_cfg.ema_tau_s,
            "drop_threshold_m": det_cfg.drop_threshold_m,
            "persistence_s": det_cfg.persistence_s,
            "r_det_at": det_cfg.r_det_at,
        },
        "resampling": {
            "dt_s": DT,
            "T": sequence.horizon_samples,
            "last_time_s": sequence.last_time_s,
            "overflow_policy": "raise_no_clipping_no_extrapolation",
            "cycle_reference_s": sequence.cycle_reference_s,
            "position": "linear_time_interpolation",
            "orientation": "SLERP_wxyz_output",
            "wrench": "uniformize_then_zero_phase_butterworth_then_interpolate",
            "wrench_anti_alias_order": ANTI_ALIAS_ORDER,
            "wrench_anti_alias_cutoff_hz": ANTI_ALIAS_CUTOFF_HZ,
        },
        "valid_run_count": len(entries),
        "success_count": n_succ,
        "failure_count": len(entries) - n_succ,
        "run_ids_in_global_tensor_order": [entry["meta"]["run_id"] for entry in entries],
    }
    with open(os.path.join(args.out_dir, "conversion_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(conversion_manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("Fertig.")


if __name__ == "__main__":
    main()
