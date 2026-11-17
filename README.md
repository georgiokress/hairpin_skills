# Hairpin-Sliding-Versuche mit UR3e

Dieses Repository enthält die ROS-2-Nodes, Parameter und Startdateien der
Masterarbeit für kraftgeregelte radiale Hairpin-Sliding-Versuche mit einem
Universal Robots UR3e und einem ATI-Kraft-/Momentensensor.

Der methodisch maßgebliche Aufnahmeweg regelt die Normalkraft mit dem
**internen UR-F/T-Sensor**. Der ATI-Sensor wird ausschließlich als passive
Referenz auf `/ati/wrench_passive` aufgezeichnet. Eine ATI→RTDE-Bridge ist für
diesen Aufnahmeweg nicht zulässig.

> **Sicherheit:** Die Programme können einen realen Industrieroboter bewegen
> und Force Mode aktivieren. Sie sind keine sicherheitsgerichtete Steuerung.
> Reale Bewegungen dürfen nur durch eingewiesene Personen am validierten
> Versuchsaufbau, bei freiem Arbeitsraum und erreichbarem Not-Halt ausgeführt
> werden. Ein Dry-Run ersetzt diese Freigabe nicht.

## Dokumentation

- [Installation und Reproduzierbarkeit](docs/INSTALLATION.md)
- [Roboter starten und Daten aufnehmen](docs/ROBOT_OPERATION.md)
- [Daten, Dateiformate und Archivierung](docs/DATA_AND_REPRODUCIBILITY.md)

## Systemstand

| Komponente | Verwendeter Stand |
| --- | --- |
| Betriebssystem | Ubuntu 24.04 LTS |
| ROS | ROS 2 Jazzy |
| Python | 3.12 |
| Roboter | Universal Robots UR3e |
| Roboter-IP | `192.168.214.203` |
| PC-/Reverse-IP | `192.168.214.204` |
| ATI Net F/T | RDT/UDP, `192.168.214.70:49152`, 500 Hz |
| UR ROS 2 Driver | Tag `3.8.0` plus Patch in `patches/` |

Die IP-Adressen und die Kalibrierung gehören zum dokumentierten Laboraufbau.
Sie dürfen nicht ungeprüft auf einen anderen Aufbau übertragen werden.

## Repository-Struktur

```text
hairpin_skills/              ROS-2-Python-Nodes
config/                      Versuchsparameter, Kalibrierung und Rosbag-QoS
launch/                      UR3e-Bring-up und statische Prozess-Frames
test/                        ament-/Python-Tests
docs/                        Installation, Betrieb und Datenmanagement
patches/                     lokaler Patch des gepinnten UR-Treibers
schemas/                     Schema des Run-Manifests
examples/                    kleine Referenz-/Beispieldateien
tools/                       bewegungsfreie Prüfwerkzeuge
hairpin_skills.repos         externe ROS-Abhängigkeiten mit Versions-Pin
package.xml                  ROS-Abhängigkeiten für rosdep
setup.py                     installierte Dateien und ROS-Entry-Points
```

Generierte Ordner (`build/`, `install/`, `log/`), Rosbags und abgeleitete
Messartefakte gehören nicht in Git.

## Schnellstart ohne Roboterbewegung

Nach der [Installation](docs/INSTALLATION.md):

```bash
source /opt/ros/jazzy/setup.bash
source ~/hairpin_ws/install/setup.bash

ros2 pkg executables hairpin_skills
colcon test --packages-select hairpin_skills
colcon test-result --verbose
```

Der folgende Aufruf führt nur die im Collector implementierte Konfigurations-
und ROS-Graph-Prüfung aus. Die mitgelieferte Konfiguration hält die beiden
physischen Freigabegates absichtlich geschlossen:

```bash
ros2 run hairpin_skills phase_b_spi_data_collection --ros-args \
  --params-file ~/hairpin_ws/src/hairpin_skills/config/phase_b_spi_data_collection.yaml
```

Dabei gelten in der YAML-Datei:

```yaml
collection_dry_run: true
run_authorized: false
```

Der komplette Ablauf vom Einschalten bis zur Aufnahme steht in
[docs/ROBOT_OPERATION.md](docs/ROBOT_OPERATION.md).

## Zentrale Programme

| Entry-Point | Zweck | Reale Bewegung |
| --- | --- | --- |
| `ati_rdt_wrench` | passiver Empfang des ATI-RDT-Datenstroms | nein |
| `phase_a_radial_ik_validator` | kontaktfreie Prüfung der radialen IK-Bahn | ja |
| `phase_b_spi_data_collection` | finaler 15-mm-Aufnahmeweg für SPI-Daten | ja |
| `phase_b_flat_reference` | kraftgeregelte Referenzfahrt auf ebener Fläche | ja |
| `phase_c_chamfer_scan` | kraftgeregelte Fasenabtastung mit Stoppkriterien | ja |
| `radial_sliding_test` | validierter radialer Sliding-Basisskill | ja |
| `force_hold_screening` | statische Kraft-Halteversuche | ja |

Weitere `test_*`- und ältere Diagnose-Entry-Points sind Entwicklungswerkzeuge
und nicht der maßgebliche Abschlussworkflow.

## Maßgeblicher Datenfluss

```text
UR interner F/T ──> Force Mode und Sicherheitsüberwachung ──┐
                                                            ├─> Rosbag (MCAP)
ATI Net F/T ──────> /ati/wrench_passive, nur Referenz ──────┘

TF + Gelenke + TCP-Pose + Wrenches + Events
                 └─> run_manifest.json ─> Offline-Auswertung
```

Die Rohdaten werden standardmäßig außerhalb des Git-Repositories unter
`/home/georgio/masterarbeit/bags` abgelegt. Eine vollständige Aufnahme enthält
zusätzlich ein unveränderliches `run_manifest.json`. Details und die sieben
Pflichttopics stehen in
[docs/DATA_AND_REPRODUCIBILITY.md](docs/DATA_AND_REPRODUCIBILITY.md).

Das Manifest kann aus jedem Clone ohne Zugriff auf den ursprünglichen
Arbeitsordner geprüft werden:

```bash
python3 tools/validate_run_manifest.py \
  /PFAD/ZUM/RUN/run_manifest.json
```

## Wissenschaftlicher Abschlussstand

Die dokumentierte Endvalidierung umfasst 40 technisch vollständige logische
Läufe aus 41 gestarteten Versuchen in vier Bedingungen. Der erste Versuch
`r026` blieb als technischer Abbruch im Roharchiv; für die Auswertung wurde nur
der Ersatzlauf `_a02_r026` verwendet. Rohdaten werden deshalb niemals allein
durch Auflisten aller Bag-Ordner ausgewählt, sondern über das versionierte
Auswahlmanifest.

Die großen MCAP-Rohdaten und Ergebnisgrafiken werden bewusst nicht in Git
gespeichert. Für eine Übergabe müssen Datensatz, Auswahlmanifest und
SHA-256-Prüfsummen gemeinsam in einem geeigneten Forschungsdatenarchiv
abgelegt werden.

## Bekannte Grenzen des übergebenen Stands

- Die ROS-Paketkompilierung ist erfolgreich.
- Die vorhandenen ament-Linttests sind derzeit nicht grün. Sie melden
  bestehende Flake8-/Pep257-Formatierungsverstöße im Forschungsquellcode; dies
  ist von einem fehlgeschlagenen Build oder Robotertest zu unterscheiden.
- Die reale Hardwareintegration ist nur für den beschriebenen Laboraufbau
  belegt. Ein Clone auf einem anderen Roboter ist keine Validierung.
- Die MCAP-Rohdaten und die vollständige Offline-SPI-Pipeline liegen aus
  Größen- und Trennungsgründen außerhalb dieses ROS-Repositories.

## Lizenz

Der eigene Quellcode steht unter der MIT-Lizenz, siehe [LICENSE](LICENSE).
Externe Abhängigkeiten behalten ihre jeweiligen Lizenzen.
