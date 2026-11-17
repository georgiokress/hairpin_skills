# Daten, Dateiformate 

## Speicherorte

Der dokumentierte Laborrechner trennt Quellcode, Rohdaten und abgeleitete
Ergebnisse:

```text
~/hairpin_ws/src/hairpin_skills/       versionierter ROS-Quellcode
~/masterarbeit/bags/phase_b_spi/       unveränderliche Rosbags
~/masterarbeit/results/phase_b_spi/    Manifeste, QC und Auswertung
~/masterarbeit/config/phase_b_spi/     Run-Pläne und Run-Konfigurationen
```

Paketressourcen werden installationsrelativ gefunden. Der Rohdatenstamm kann
mit `HAIRPIN_DATA_ROOT` beziehungsweise dem ROS-Parameter `bag_root` gewählt
werden. Die Offline-SPI-Skripte verwenden `HAIRPIN_DATASET_ROOT` und
`HAIRPIN_RESULTS_ROOT`; siehe `SPI_WORKFLOW.md`.

## Inhalt eines Rohdaten-Runs

```text
<run_id>/
├── <run_id>_0.mcap
├── metadata.yaml
└── run_manifest.json
```

`run_manifest.json` enthält unter anderem Identität, Versuchszweck,
Sollparameter, verwendete Frames, Startpose, Software-/Konfigurationshashes,
Ground Truth, Bag-Pfad und Abschlussstatus. Rohbag und ursprüngliches Manifest
werden nach Abschluss nicht mehr verändert.

## Pflichttopics

Ein vollständiger Phase-B-SPI-Run enthält nichtleere Nachrichten auf:

- `/tf`
- `/tf_static`
- `/joint_states`
- `/tcp_pose_broadcaster/pose`
- `/force_torque_sensor_broadcaster/wrench`
- `/ati/wrench_passive`
- `/phase_b_spi_collection/events`

Der interne UR-Wrench ist die einzige Wrench-Repräsentation der
15-dimensionalen SPI-Trajektorie. ATI bleibt eine getrennte passive Referenz.

## Segment und Kanäle

Die auszuwertende Sliding-Trajektorie beginnt beim Event `slide_started` nach
Touchdown und Kontaktstabilisierung. Anfahrt, Tare, Touchdown und Rückfahrt
gehören nicht zum Modellsegment.

| Index | Kanal | Einheit/Konvention |
| ---: | --- | --- |
| 0 | `p_EOS` | 0 vor Ende, 1 ab Endsample |
| 1 | `p_succ` | konstantes Run-Label |
| 2–4 | `x_R, y_R, z_R` | m im `sliding_frame` |
| 5–8 | `qw, qx, qy, qz` | normiert, w-first |
| 9–11 | `Fx_R, Fy_R, Fz_R` | N, interner UR-Sensor |
| 12–14 | `Mx_R, My_R, Mz_R` | Nm, interner UR-Sensor |

Kompression wird im Modell als positiver Betrag behandelt. Das ROS-Kommando
für die validierte Richtung ist negativ:

```text
normal_force_n = -F_N_mag
```

## Run-ID und Schutz vor Überschreiben

Der Collector erzwingt für den Basisworkflow das Schema:

```text
pb_spi_s4_setznest2_intctrl_atipass_d015_v####_fn###_<purpose>_r###
```

`v####` kodiert `radial_speed_mps * 10000`, `fn###` den Kraftbetrag in
Zehntel-Newton. Run-ID, YAML-Parameter und Zielpfad müssen übereinstimmen.
Existierende Bag-Ordner dürfen nicht überschrieben oder umbenannt werden.

