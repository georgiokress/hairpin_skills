"""Report existing searches at matched time weights. No training, test or robot I/O."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def compare(rows):
    result = []
    for tw in sorted({r['time_weight'] for r in rows}):
        base = None
        for ew in sorted({r['error_weight'] for r in rows}):
            group = [r for r in rows if r['time_weight'] == tw and r['error_weight'] == ew]
            if len(group) != 3:
                raise ValueError('Expected three preserved starts per weight pair')
            for r in group:
                expected = (1 - r['soft_success_score'] + tw * r['soft_time_s'] / 8.192
                            + ew * r['error_mm'] / 1.5)
                if not math.isclose(expected, r['best_objective'], abs_tol=2e-6, rel_tol=2e-6):
                    raise ValueError('Stored objective inconsistent with terms')
            feasible = [r for r in group if r['model_feasible']]
            if not feasible:
                raise ValueError('No feasible candidate; report explicitly before selecting')
            winner = min(feasible, key=lambda r: (r['best_objective'], r['candidate_id']))
            if ew == 0:
                base = winner
            if base is None:
                raise ValueError('Missing zero-error baseline')
            result.append(dict(winner=winner, compared_candidate_ids=[r['candidate_id'] for r in group],
                               baseline_candidate_id=base['candidate_id'],
                               delta_time_s=winner['time_s']-base['time_s'],
                               delta_error_mm=winner['error_mm']-base['error_mm'],
                               delta_force_N=winner['force_N']-base['force_N'],
                               delta_v_mm_s=winner['v_mm_s']-base['v_mm_s']))
    return result


def run(source, output):
    rows = json.loads(source.read_text())
    result = compare(rows)
    if len(rows) != 60 or len(result) != 20:
        raise ValueError('Expected complete 60-search, 20-pair comparison')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='Development comparison after search; not preregistered or independent validation',
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  selection='Minimum stored objective among three model-feasible searches for each weight pair; tie by candidate ID',
                  same_model_and_context=True, independent_test_access=False, robot_motion=False,
                  final_weights_selected=False, comparisons=result)
    (output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Ursprüngliche Güte und Fehler-Erweiterung: Vergleich bei gleichem Zeitgewicht', '',
             'Nutzerentscheidung: ursprüngliche Success–Zeit-Güte als Hauptauswertung und Fehler-Erweiterung als ergänzende Untersuchung verwenden. Noch keine endgültige Gewichtswahl oder Erweiterung der physischen Runzahl.', '',
             'Gleich bleiben Modell, Daten, Startkontext, Parametergrenzen, Endpunktbehandlung und Suchbudget. Je Gewichtspaar wird die kleinste Zielfunktion aus drei modellbasiert zulässigen Suchen ausgewählt. Unterschiede zur Basis beziehen sich immer auf dasselbe Zeitgewicht. Zielfunktionswerte unterschiedlicher Gewichte werden nicht miteinander gerankt.', '',
             'Alle Zeiten und Fehler unten sind Modellvorhersagen. EOS ist virtuelle Detektionsdauer, nicht volle Roboterzykluszeit. Die Auswahlregel wurde nach der ursprünglichen Suche dokumentiert; dies ist keine Vorregistrierung.', '',
             '| Zeitgewicht | Fehlergewicht | ID | Kraft [N] | v [mm/s] | EOS [s] | Fehler [mm] | Δ EOS [s] | Δ Fehler [mm] |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result:
        w = r['winner']
        lines.append(f"| {w['time_weight']:g} | {w['error_weight']:g} | {w['candidate_id']} | {w['force_N']:.3f} | {w['v_mm_s']:.3f} | {w['time_s']:.3f} | {w['error_mm']:.3f} | {r['delta_time_s']:+.3f} | {r['delta_error_mm']:+.3f} |")
    lines += ['', '## Einordnung', '',
              '- Bei Zeitgewicht 1 bleibt die beste gefundene Lösung für Fehlergewichte 0; 0,3; 1; 3 bei 1 N/10 mm/s. Die Erweiterung erzwingt keine andere Lösung.',
              '- Bei Zeitgewicht 0,1 verschiebt sich die Lösung mit höherem Fehlergewicht zu langsameren Einstellungen. Beide Terme sind daher als gemeinsamer Präferenzraum zu berichten.',
              '- Die zuvor gezeigten Punkte bei etwa 3,97 und 5,50 mm/s sind gefundene Pareto-Kandidaten, aber nicht die kleinsten Zielfunktionswerte ihrer drei Starts bei Zeitgewicht 0,3/Fehlergewicht 1. Sie bleiben transparent erhalten und dürfen nicht als beste Multistart-Ergebnisse bezeichnet werden.',
              '- Bei 1 N/10 mm/s stehen 0,275 mm vorhergesagtem Fehler bereits 0,677 mm gemessener Development-Mittelwert gegenüber. Eine Verbesserung im Modell ist noch keine reale Verbesserung.',
              '- Modell, Success-Label und Detektor werden für den Gütevergleich nicht verändert. Der unabhängige Modelltest wird nicht zur Auswahl günstiger Gewichte verwendet.',
              '- Der reale Plan bleibt zunächst bei 60 Läufen. Zwei unterschiedliche SPI-Kandidaten zusätzlich zu Referenz und Best-of-DOE ergäben bei 20 Blöcken vier Bedingungen und 80 Läufe; eine solche Erweiterung ist gesondert zu entscheiden.',
              '', f"Quelle: {source}", f"SHA256: {report['source_sha256']}", '']
    (output/'Vergleich_Guetevarianten.md').write_text('\n'.join(lines))
    print(json.dumps(dict(comparisons=len(result), searches=len(rows), output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    run(args.source, args.output)
