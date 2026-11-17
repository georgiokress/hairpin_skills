# SPI ausführen, trainieren und invertieren

Diese Anleitung beschreibt die tatsächlich verwendete Offline-Kette. Alle
Befehle sind bewegungsfrei. Roboterdaten werden nur gelesen; eine Inversion
liefert Kandidaten, sie startet keinen realen Versuch.

## 1. Python-Umgebung

```bash
cd ~/hairpin_ws/src/hairpin_skills
python3 -m venv .venv-spi
source .venv-spi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r spi/requirements.txt
```

Der dokumentierte Abschlussstand verwendete Python 3.11 sowie NumPy 2.4.6,
SciPy 1.17.1, PyTables 3.11.1, PyTorch 2.13.0 (CPU) und Matplotlib 3.11.1.


## 2. Rohbags exportieren

Der finale Exporter liest nur technisch abgeschlossene MCAP-Bags und die getrennten
Operatorannotationen. Er segmentiert von `slide_started` bis
`slide_finished`, transformiert TCP-Pose und internen UR-Wrench ins
`sliding_frame` und prüft, dass ATI ausschließlich passiv vorhanden ist.

Da `rosbag2_py` und ROS-Nachrichtentypen benötigt werden, erfolgt dieser
Schritt in einem ROS-Jazzy-Terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hairpin_ws/install/setup.bash

export HAIRPIN_MAIN_CONFIG_ROOT=/PFAD/ZU/DEN/RUN_KONTEXTEN
python3 data_conversion/extract_native.py \
  /PFAD/ZUM/source_acquisition_inventory.json \
  /PFAD/ZUM/NEUEN/EXPORTORDNER
```

Der Zielordner darf noch nicht existieren. Das Inventar verweist auf die Bags;
die Operatorannotationen liegen neben dem Inventar in `operator_annotations/`.
Pro Run entstehen `meta.json` und `series.npz`. Die Hauptserie umfasst exakt
90 gültige Runs. Falls im historischen Export nur ergänzende Kontextdateien
fehlten, kann danach die streng begrenzte Rekonziliation laufen:

```bash
export HAIRPIN_MAIN_RUN_PLAN=/PFAD/ZUM/run_plan_v1.json
export HAIRPIN_MAIN_RESULTS_ROOT=/PFAD/ZU/results/phase_b_spi/main_doe_v1
python3 data_conversion/reconcile_native.py /PFAD/ZUM/NEUEN/EXPORTORDNER
```

Anschließend erzeugt die SPI-Umgebung Labels, HDF5-Bundle und gruppierte
Fold-Mitgliedschaften:

```bash
source .venv-spi/bin/activate
export HAIRPIN_MAIN_BAG_ROOT=/PFAD/ZU/bags/phase_b_spi/main_doe_v1
python data_conversion/analyze_export.py /PFAD/ZUM/NEUEN/EXPORTORDNER
```

`reconcile_native.py` ergänzt niemals Messwerte. Es darf ausschließlich
belegbaren Kontext aus dem eingefrorenen Plan ergänzen. Die 27 Pilotläufe
wurden mit dem in `data_conversion/spi_bounds_v1/` archivierten Exportvertrag
aufbereitet und anschließend unverändert in das 117-Run-Development-Bundle
übernommen.

Die für das Training erwartete Struktur lautet:

```text
datasets/phase_b_spi/
├── main_doe_v1_spi_v1/
│   ├── full/train/main_doe_all.h5
│   ├── grouped_cv_membership_v1.json
│   ├── offline_export_summary_v1.json
│   ├── native_runs/...
│   └── evaluation/labels.json
└── development_main90_pilot27_v1/
    ├── full/train/development_all.h5
    ├── grouped_cv_membership_v1.json
    └── development_bundle_manifest_v1.json
```

Falls das kombinierte 117-Run-Bundle noch nicht vorliegt, wird es aus dem
eingefrorenen Haupt- und Pilotexport erstellt:

```bash
export HAIRPIN_DATASET_ROOT=/ABSOLUTER/PFAD/datasets/phase_b_spi
export HAIRPIN_PILOT_ANNOTATIONS=/PFAD/ZU/pilot_doe_v0/operator_annotations
python spi/include_pilot.py \
  "$HAIRPIN_DATASET_ROOT/development_main90_pilot27_v1"
```

Der Zielordner muss neu sein. Das Skript prüft Detektor-, Quellcode-, Bag-,
Manifest- und Annotationshashes und verändert die ursprünglichen Exporte nicht.

Die HDF5-Dateien enthalten `N × 640 × 15` Trajektorien. Der Loader akzeptiert
nur als `development` markierte Daten und prüft Detektorhash, Run-IDs, Shapes,
Masken, EOS und Fold-Trennung. Die großen Datensätze selbst gehören nicht in
Git und müssen separat mit Hashliste übergeben werden.

## 3. Pfade setzen

```bash
cd ~/hairpin_ws/src/hairpin_skills
source .venv-spi/bin/activate
export HAIRPIN_DATASET_ROOT=/ABSOLUTER/PFAD/datasets/phase_b_spi
export HAIRPIN_RESULTS_ROOT=/ABSOLUTER/PFAD/results/phase_b_spi
mkdir -p "$HAIRPIN_RESULTS_ROOT"
```

Jeder Ergebnisordner muss neu sein. Die Skripte überschreiben keine frühere
Studie.

## 4. Qualitätszweig trainieren

Zuerst wird der separate Ortsfehler-Ausgang in fünf gruppierten Folds und drei
festen Seeds trainiert. Beide Datenvarianten erhalten je 4000 Updates.

```bash
python spi/compare_quality_cv_v2.py \
  "$HAIRPIN_RESULTS_ROOT/pilot_benefit_quality_cv_v2"
```

Wesentliche Ausgaben sind `comparison_summary.json`, die Fold-Checkpoints und
`protocol_frozen_before_training.json`.

## 5. Trajektorienmodell evaluieren

Die vollständige Development-CV lädt den eingefrorenen Qualitätszweig und
trainiert EOS, Success und die 15-kanalige Residualtrajektorie in fünf Folds
und drei Seeds. Das feste Budget beträgt 50 Epochen.

Zunächst wird der historische Fold-1-Reproduktionslauf erzeugt:

```bash
python spi/train_trajectory_trial_v1.py \
  "$HAIRPIN_RESULTS_ROOT/trajectory_development_trial_v1"
```

Danach die vollständige CV:

```bash
python spi/train_trajectory_cv_v1.py \
  "$HAIRPIN_RESULTS_ROOT/trajectory_development_cv_v1" \
  --workers 3
```

Optional wird der harte EOS-Schwellwert ausschließlich aus den jeweiligen
Trainingsfolds bestimmt und Out-of-Fold ausgewertet:

```bash
python spi/calibrate_endpoint_v1.py \
  "$HAIRPIN_RESULTS_ROOT/endpoint_calibration_v1"
```

Diese Checkpoints sind Development-Auswertung, noch kein finales Refit.

## 6. Modell auf allen 117 Development-Läufen refitten

Der in der Arbeit verwendete Kandidat nutzt 90 Haupt- plus 27 Pilotläufe,
Seed `20260918`, 4000 Quality-Schritte und 300 Trajektorienupdates:

```bash
python spi/refit_candidate_v1.py \
  "$HAIRPIN_RESULTS_ROOT/refit_main_plus_pilot_v1" \
  --variant main_plus_pilot
```

Der Checkpoint heißt `REFIT_CANDIDATE_NOT_TEST_RELEASED.pt`. Das Skript prüft
nach dem Speichern eine bitgenaue Reload-Vorhersage. Der historische
Checkpoint hatte SHA-256
`79b2a744f136acd650a6446a0fb1354a05df1e9e4a06d10472e7c72cf40e1f56`;
ein neuer Lauf darf wegen Plattform-/Bibliotheksunterschieden abweichen und
muss mit seinem eigenen Protokollhash berichtet werden.

## 7. SPI-Inversion ausführen

```bash
python spi/invert_candidate_pareto_v1.py \
  "$HAIRPIN_RESULTS_ROOT/refit_main_plus_pilot_v1" \
  "$HAIRPIN_RESULTS_ROOT/inversion_pareto_v1"
```

Die Inversion optimiert differentiabel durch analytischen Sliding-Prior,
Residual-GRU und Soft-Endpoint. Sie verwendet:

- Grenzen: 1–10 mm/s und 1–5 N;
- Zeitgewichte: 0,1; 0,3; 1; 3; 10;
- Fehlergewichte: 0; 0,3; 1; 3;
- drei Startpunkte: (4 mm/s, 3 N), (2 mm/s, 5 N), (10 mm/s, 1 N);
- 60 Suchen mit jeweils 120 Adam-Schritten.

Die Zielfunktion ist

```text
J = 1 - S_soft + lambda_t * t_soft / 8.192 s
                  + lambda_e * e_abs,pred / 1.5 mm
```

Anschließend lassen sich die beiden Zielfunktionsfamilien bei identischem
Zeitgewicht vergleichen:

```bash
python spi/compare_objective_families_v1.py \
  "$HAIRPIN_RESULTS_ROOT/inversion_pareto_v1/candidates.json" \
  "$HAIRPIN_RESULTS_ROOT/objective_comparison_v1"
```

Der in der Arbeit ausgewählte Vergleich verwendete `lambda_t = 0.1`. Daraus
ergaben sich als SPI-Kandidaten 1 N / 7.6067 mm/s ohne zusätzlichen
Fehlerterm und 1 N / 2.2461 mm/s mit Fehlergewicht 1. Diese Werte sind
Modellkandidaten; eine reale Ausführung benötigt weiterhin den vollständigen
Freigabe- und Aufnahmeablauf aus `ROBOT_OPERATION.md`.

## 8. Reproduzierbarkeit berichten

Für jeden Lauf sind mindestens zu sichern:

- Git-Commit dieses Repositories;
- `protocol_*.json` und Quellcodehashes;
- HDF5-, Membership- und Checkpoint-SHA-256;
- Python- und Paketversionen;
- Ergebnisordner der CV, des Refits und der Inversion;
- die Information, dass keine Testdaten zur Modellauswahl verwendet wurden.
