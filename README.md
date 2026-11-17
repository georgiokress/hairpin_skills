# Hairpin-Sliding- und SPI-Workflow

Dieses Repository enthält ausschließlich den für die Abschlussarbeit relevanten
Endstand: den kraftgeregelten 15-mm-Sliding-Skill, die reproduzierbare
Datenaufnahme und die Offline-Pipeline für Training und Inversion der Shadow
Program Inversion (SPI).

Der Roboter regelt die Normalkraft mit dem **internen UR-F/T-Sensor**. Der
ATI-Sensor wird nur passiv auf `/ati/wrench_passive` aufgezeichnet und hat
keinen Einfluss auf die Regelung.

## Was tatsächlich verwendet wurde

```text
UR3e + interner F/T ──> phase_b_spi_data_collection ──> MCAP + Manifest
ATI Net F/T ──────────> ati_rdt_wrench (nur passive Referenz) ─┘
                                      │
                                      v
data_conversion/export_rosbags.py ──> native Runs / Datensatz
                                      │
                                      v
spi/  ──> Qualitätsmodell + Trajektorienmodell ──> Refit ──> Inversion
```

Phase A, Phase C sowie frühere Diagnose- und Testprogramme waren
Entwicklungsstände und gehören nicht zum finalen wissenschaftlichen Ablauf.
Sie bleiben über die Git-Historie nachvollziehbar, stehen aber nicht mehr im
aktuellen Arbeitsstand.

## Einstieg

- [Installation](docs/INSTALLATION.md)
- [Roboter starten und Daten aufnehmen](docs/ROBOT_OPERATION.md)
- [SPI-Daten aufbereiten, Modell trainieren und invertieren](docs/SPI_WORKFLOW.md)
- [Datenformat und Reproduzierbarkeit](docs/DATA_AND_REPRODUCIBILITY.md)

Bewegungsfreier ROS-Preflight:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hairpin_ws/install/setup.bash
ros2 run hairpin_skills phase_b_spi_data_collection --ros-args \
  --params-file ~/hairpin_ws/src/hairpin_skills/config/phase_b_spi_data_collection.yaml \
  -p manifest_schema_path:=$HOME/hairpin_ws/src/hairpin_skills/schemas/run_manifest.schema.json \
  -p ground_truth_file:=$HOME/hairpin_ws/src/hairpin_skills/examples/rgt_caliper_v1.json
```

Die mitgelieferte Konfiguration setzt `collection_dry_run: true` und
`run_authorized: false`. Damit findet keine Bewegung statt.

Bewegungsfreier SPI-Test:

```bash
cd ~/hairpin_ws/src/hairpin_skills
python3 -m venv .venv-spi
source .venv-spi/bin/activate
pip install -r spi/requirements.txt
python -m unittest discover -s spi -p 'test_*.py'
```

## Zentrale Dateien

| Datei | Zweck |
| --- | --- |
| `hairpin_skills/radial_sliding_test.py` | validierter Sliding-Basisskill |
| `hairpin_skills/phase_b_spi_data_collection.py` | finaler Collector mit Gates, Events, Rosbag und Manifest |
| `hairpin_skills/ati_rdt_wrench.py` | passiver ATI-RDT-Empfänger |
| `launch/ur3e_bringup.launch.py` | UR3e-Treiber und Prozessframes |
| `config/phase_b_spi_data_collection.yaml` | sichere Collector-Vorlage für 15 mm |
| `data_conversion/extract_native.py` | die 90 Hauptserien-Bags und Annotationen exportieren |
| `spi/refit_candidate_v1.py` | finales Development-Refit |
| `spi/invert_candidate_pareto_v1.py` | beschränkte Multi-Start-Inversion |

## Wissenschaftlicher Stand

Das Development-Modell wurde auf 117 Läufen trainiert (90 Haupt- und 27
Pilotläufe). Die Inversion suchte innerhalb von 1–5 N und 1–10 mm/s über 20
Gewichtspaare und drei feste Startpunkte, also 60 Optimierungen mit je 120
Adam-Schritten. Die real ausgewertete Endvalidierung umfasst 40 vollständige
logische Läufe; der technisch abgebrochene erste Versuch `r026` blieb im
Roharchiv und wurde durch `_a02_r026` ersetzt.

Rohbags und große Ergebnisartefakte sind nicht Bestandteil von Git. Sie
müssen zusammen mit Auswahlmanifest und SHA-256-Prüfsummen separat archiviert
werden.

## Lizenz

Eigener Quellcode: MIT, siehe [LICENSE](LICENSE). Externe Abhängigkeiten
behalten ihre jeweilige Lizenz.
