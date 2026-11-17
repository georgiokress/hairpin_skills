"""
Eingefrorener Pilotdetektor (pilot_doe_v0), positionsbasiert auf z_R.

Spezifikation (pilot_doe_v0.md, Stand 26.08.2026):
- lineare Baseline z_R(x_R) bis 7,5 mm Verschiebung;
- frühestmögliche Detektion ab 7,5 mm;
- kausales EMA mit Zeitkonstante 32 ms (auf nativen ROS-Zeitstempeln,
  VOR dem 32-ms-Resampling);
- Abfallschwelle 0,075 mm (EMA-Residuum <= -0,075 mm);
- Persistenz 32 ms (Bedingung muss 32 ms ununterbrochen gelten).

Konventionen dieses Moduls:
- t: native Zeitstempel in Sekunden, t=0 beim Ereignis slide_started.
- d: Verschiebung in +x_R ab slide_started, in Metern.
- z: z_R in Metern (sliding_frame).
- t_cross: erster Zeitstempel der persistenten Episode (erste Unterschreitung).
- t_det:   Zeitstempel, an dem die 32-ms-Persistenz erfüllt ist (kausaler
           Online-Triggerzeitpunkt; dies ist der EOS-relevante Zeitpunkt).
- r_det:   Verschiebung d zum Zeitpunkt t_det (Umschalten auf t_cross über
           r_det_at="cross" möglich; Wahl dokumentieren und konsistent halten).
"""
from dataclasses import dataclass

import numpy as np

MM = 1e-3


@dataclass(frozen=True)
class PilotDetectorConfig:
    baseline_max_disp_m: float = 7.5 * MM   # lineare Baseline bis 7,5 mm
    min_detect_disp_m: float = 7.5 * MM     # Detektion erst ab 7,5 mm
    ema_tau_s: float = 0.032                # kausales EMA, 32 ms
    drop_threshold_m: float = 0.075 * MM    # Abfallschwelle 0,075 mm
    persistence_s: float = 0.032            # Persistenz 32 ms
    r_det_at: str = "det"                   # "det" (Persistenz erfüllt) | "cross"


@dataclass
class DetectionResult:
    triggered: bool
    t_cross: float          # s, NaN wenn kein Trigger
    t_det: float            # s, NaN wenn kein Trigger
    r_det_m: float          # m, NaN wenn kein Trigger
    baseline_coeffs: tuple  # (Steigung m/m, Achsenabschnitt m) der z_R(x_R)-Gerade
    residual_ema: np.ndarray  # EMA-Residuum je nativem Sample (m), für QC-Plots


def causal_ema_irregular(t: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    """Kausales EMA erster Ordnung auf unregelmäßigen Zeitstempeln."""
    y = np.empty_like(x)
    y[0] = x[0]
    dt = np.diff(t)
    alpha = 1.0 - np.exp(-np.maximum(dt, 0.0) / tau)
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha[i - 1] * (x[i] - y[i - 1])
    return y


def run_pilot_detector(t: np.ndarray, d: np.ndarray, z: np.ndarray,
                       cfg: PilotDetectorConfig = PilotDetectorConfig()) -> DetectionResult:
    """
    t, d, z: native Samples ab slide_started (t[0] >= 0, d monoton wachsend ~ v*t).
    Rückgabe siehe DetectionResult.
    """
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    assert t.shape == d.shape == z.shape and t.ndim == 1 and len(t) >= 10

    # 1) Lineare Baseline z_R(x_R) auf d <= 7,5 mm
    mask = d <= cfg.baseline_max_disp_m
    if mask.sum() < 5:
        raise ValueError("Zu wenige Samples im Baseline-Fenster (d <= 7,5 mm)")
    slope, intercept = np.polyfit(d[mask], z[mask], 1)
    baseline = slope * d + intercept

    # 2) Residuum + kausales EMA (native Stempel, vor Resampling)
    residual = z - baseline
    residual_ema = causal_ema_irregular(t, residual, cfg.ema_tau_s)

    # 3) Schwelle + Persistenz, frühestens ab 7,5 mm Verschiebung
    below = (residual_ema <= -cfg.drop_threshold_m) & (d >= cfg.min_detect_disp_m)

    t_cross = np.nan
    t_det = np.nan
    triggered = False
    episode_start = None
    for i in range(len(t)):
        if below[i]:
            if episode_start is None:
                episode_start = i
            if t[i] - t[episode_start] >= cfg.persistence_s:
                t_cross = t[episode_start]
                t_det = t[i]
                triggered = True
                break
        else:
            episode_start = None

    if triggered:
        t_r = t_det if cfg.r_det_at == "det" else t_cross
        r_det = float(np.interp(t_r, t, d))
    else:
        r_det = np.nan

    return DetectionResult(triggered=triggered, t_cross=float(t_cross),
                           t_det=float(t_det), r_det_m=r_det,
                           baseline_coeffs=(float(slope), float(intercept)),
                           residual_ema=residual_ema)
