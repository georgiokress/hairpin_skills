# Roboter starten und Daten aufnehmen

Diese Anleitung beschreibt den finalen Aufnahmeweg der Masterarbeit: interne
UR-Kraftregelung, ATI nur passiv, 15-mm-Bahn in `+x_R`.


## 1. Checkliste vor dem Start

- UR3e, Greifer, Hairpin, Setznest, TCP und Payload entsprechen dem validierten
  Aufbau.
- Die roboterspezifische Kalibrierung und das `sliding_frame` sind unverändert.
- Der Arbeitsraum ist frei; Schutzmaßnahmen und Not-Halt sind funktionsbereit.
- Roboter und PC befinden sich im Netz `192.168.214.0/24`.
- Der Roboter ist unter `192.168.214.203`, der Steuer-PC unter
  `192.168.214.204` erreichbar.
- Am Pendant ist das validierte External-Control-Programm geladen, aber noch
  nicht automatisch gestartet.
- Der ATI-URCap-Daemon ist gestoppt. Es laufen weder ATI→RTDE-Bridge noch
  `ATI_RTDE_enable`/`ATI_RTDE_disable`.

## 2. Terminal vorbereiten

In jedem Terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hairpin_ws/install/setup.bash
```

## 3. Passiven ATI-Datenstrom starten

Terminal 1:

```bash
ros2 run hairpin_skills ati_rdt_wrench --ros-args \
  -p standalone_mode_confirmed:=true \
  -p topic:=/ati/wrench_passive
```

Dieser Node bewegt den Roboter nicht. `standalone_mode_confirmed:=true` darf
nur gesetzt werden, wenn der ATI-URCap-Daemon tatsächlich gestoppt ist. Der
erwartete Frame ist `ati_ur_aligned`.

Lesende Prüfung in einem weiteren Terminal:

```bash
ros2 topic echo --once /ati/wrench_passive
ros2 topic hz /ati/wrench_passive
```

## 4. UR-Treiber starten

Terminal 2:

```bash
ros2 launch hairpin_skills ur3e_bringup.launch.py \
  robot_ip:=192.168.214.203 \
  reverse_ip:=192.168.214.204 \
  headless_mode:=false \
  launch_rviz:=false
```

Wichtige Eigenschaften dieses Aufrufs:

- `headless_mode:=false`: Das Pendant-Programm wird nicht automatisch
  gestartet.
- Der finale Bring-up enthält keine externe F/T-RTDE-Aktivierung. Die
  Kraftregelung verwendet ausschließlich den internen UR-Sensor.
- Der Bring-up lädt den Treiber, die Controller, die Kalibrierung und die
  statischen Frames. Er erteilt selbst keinen Trajektorienauftrag.

Erst nach erfolgreichem Treiberstart startet die Bedienperson das
External-Control-Programm am Pendant.

## 5. Bereitschaft lesend prüfen

Diese Prüfungen erteilen keinen Bewegungsauftrag:

```bash
ros2 control list_controllers
ros2 topic echo --once /io_and_status_controller/robot_program_running
ros2 topic echo --once /force_torque_sensor_broadcaster/wrench
ros2 topic hz /force_torque_sensor_broadcaster/wrench
ros2 run tf2_ros tf2_echo sliding_frame tool0_controller
```

Erwarteter Controllerzustand vor der Aufnahme:

| Controller | Zustand |
| --- | --- |
| `scaled_joint_trajectory_controller` | `active` |
| `force_mode_controller` | `inactive` |
| `passthrough_trajectory_controller` | `inactive` |
| `freedrive_mode_controller` | `inactive` |

Zusätzlich müssen internes Wrench-Signal und passives ATI-Signal aktuell,
endlich und im erwarteten Frame sein. Ein numerisch festhängendes Nullsignal
ist ein Abbruchgrund.

## 6. Bewegungsfreien Collector-Preflight ausführen

Die mitgelieferte YAML-Datei ist absichtlich nicht für einen physischen Lauf
freigeschaltet:

```bash
ros2 run hairpin_skills phase_b_spi_data_collection --ros-args \
  --params-file ~/hairpin_ws/src/hairpin_skills/config/phase_b_spi_data_collection.yaml
```

Der Preflight prüft unter anderem:

- Parameter, Dateipfade und Run-ID;
- internes UR-Wrench-Topic und passives ATI-Topic;
- Abwesenheit verbotener ATI→RTDE-Komponenten;
- benötigte Actions und Services;
- mindestens 20 endliche interne Samples in zwei Sekunden;
- TF-Kette `tool0_controller -> sliding_frame` und den kalibrierten
  `base -> sliding_frame`-Wert.

Er startet weder Force Mode noch Tare, Bag-Aufzeichnung oder Bewegung.

## 7. Run-Konfiguration vorbereiten

Für jeden realen Lauf wird eine eigene, unveränderliche YAML-Datei verwendet.
Mindestens folgende Felder müssen untereinander konsistent sein:

- `bag_name`: eindeutige Run-ID;
- `bag_root`: Zielverzeichnis der Rohdaten;
- `radial_speed_mps` und das zugehörige `v....`-Token der Run-ID;
- `normal_force_n` (negatives ROS-Kommando) und das zugehörige `fn...`-Token;
- `purpose`, `planned_split`, `repetition_index`, `planned_order_index`;
- `ground_truth_file`, `manifest_schema_path` und `repository_path` verwenden
  normalerweise die installierten beziehungsweise aus dem Paket ermittelten
  Standardwerte;
- `collection_config_path` zeigt bei einem realen Lauf auf genau dessen
  unveränderliche YAML-Datei;
- `collection_dry_run: true` und `run_authorized: false` bleiben in der
  gespeicherten Datei geschlossen.

Die physische Freigabe wird nur für den einzelnen Prozessaufruf überschrieben.
Die YAML-Datei selbst bleibt als Provenienzdatei unverändert.

## 8. Einen ausdrücklich freigegebenen Lauf aufnehmen

Nur nach erfolgreicher Checkliste, aktuellem Preflight und ausdrücklicher
Freigabe der Bedienperson:

```bash
ros2 run hairpin_skills phase_b_spi_data_collection --ros-args \
  --params-file /ABSOLUTER/PFAD/ZUR/RUN_KONFIGURATION.yaml \
  -p collection_dry_run:=false \
  -p run_authorized:=true
```

Der Collector verlangt danach zwei Texteingaben:

1. `RUN` vor jeder Controllerinteraktion beziehungsweise Bewegung;
2. `ACQUIRE` vor Rosbag, internem Tare und Force Mode.

Der normale Ablauf ist:

1. Vorbedingungen und Startpose prüfen;
2. internen Sensor taren und kontaktfreie Baseline erfassen;
3. Kontakt aufbauen und stabilisieren;
4. 15 mm in `+x_R` fahren und gleichzeitig Daten aufzeichnen;
5. Force Mode stoppen;
6. kontrolliert zur validierten Startpose zurückfahren;
7. Rosbag und `run_manifest.json` abschließen.

`Ctrl+C` wird an den aktuellen Prozess weitergegeben, ist aber **kein**
sicherheitsgerichteter Not-Halt.



## Abbruchkriterien

Der Lauf wird nicht gestartet beziehungsweise sofort sicher beendet, wenn
unter anderem einer der folgenden Punkte vorliegt:

- Safety-Zustand nicht `NORMAL` oder External Control nicht verbunden;
- Startpose außerhalb der validierten Toleranz;
- fehlender/falscher TF oder unerwarteter Frame;
- internes oder passives Wrench-Signal fehlt, ist veraltet, nicht endlich oder
  numerisch festgefroren;
- unerwarteter Controller aktiv;
- ATI→RTDE-Komponente im ROS-Graph aktiv;
- Bag- oder Ergebnisordner existiert bereits;
- Aufbau, Hairpin, TCP, Payload oder Freigabesituation ist unklar.
