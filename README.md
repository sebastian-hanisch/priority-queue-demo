# Prioritätsklassen – wer darf vor? (Streamlit-Demo)

**[→ Demo live ausprobieren](https://sebastianhanisch-priority-queue-demo.streamlit.app/)**

---

Interaktive Demo zur **Vorfahrt für eilige Lkw** am Terminal-Gate. **Elftes Stück der Konzepte-Linie „Warteschlangentheorie und Simulation“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Bisher wurden alle Lkw gleich behandelt. Hier gibt es **Eilige** (Klasse 1: Express, Kühlcontainer, enge Fristen) und **Standard-Lkw** (Klasse 2). Vorfahrt macht die Wartezeit der Eiligen drastisch kürzer,
aber die Wartezeit verschwindet nicht: sie wandert zu den anderen (**Erhaltungssatz**). Die Formeln liefern [mg1-kingman-demo](https://github.com/sebastian-hanisch/mg1-kingman-demo) (Stück 10, eine Klasse)
und [mmc-queue-demo](https://github.com/sebastian-hanisch/mmc-queue-demo) (Stück 3, mehrere Spuren); neu ist die **exakt gelöste zweidimensionale Markov-Kette** als unabhängige Referenz.

## Kernfrage

Wie viel gewinnen die Eiligen, wie viel zahlen die anderen, wann lohnt sich Unterbrechen, und wie viele Eilige verträgt die Vorfahrt?

## Modell und Methodik

- **Gate:** c Spuren (1 oder 4), Poisson-Ankünfte, Auslastung ρ je Spur (50 bis 95 %), Anteil Eiliger p (10 bis 50 %), **gleiche Abfertigungsdauer** für beide Klassen (Mittel 3 min, Streuung cs² = 0, 1 oder 4).
  **Verzögerung** = Zeit im System − Dauer (bei Unterbrechung inklusive der Unterbrechungen). Drei Reihenfolgen: **FIFO**, **Vorfahrt nicht unterbrechend**, **Vorfahrt unterbrechend** (mit Wiederaufnahme).
- **Formeln** (`prio_formulas.py`): Eine Spur, jede Dauer: nicht unterbrechend nach **Cobham**: W₀ = ρ(1 + cs²)/2, W₁ = W₀/(1 − σ₁), W₂ = W₀/((1 − σ₁)(1 − σ₂)), σ₁ = pρ, σ₂ = ρ; unterbrechend
  (Preemptive-Resume): Tₖ = 1/(1 − σₖ₋₁) + Rₖ/((1 − σₖ₋₁)(1 − σₖ)), Verzögerung Tₖ − 1. Mehrere Spuren, exponentielle Dauer: Erlang C mit dem Cobham-Faktor; unterbrechend sieht Klasse 1 ein eigenes M/M/c
  mit Auslastung pρ, das Gesamtmittel ist das von M/M/c, Klasse 2 folgt aus dem Erhaltungssatz. Für mehrere Spuren mit anderer Dauer gibt es keine Formel (die App zeigt dann nur die Simulation).
- **Markov-Kette** (`prio_chain.py`): Bei exponentieller Dauer ist der Zustand (n₁, n₂) = Zahl der Eiligen und der Standard-Lkw im System (bei nicht unterbrechender Vorfahrt und einer Spur zusätzlich die Klasse in
  Abfertigung). Das Gleichgewicht πQ = 0 wird als dünn besetztes lineares System über alle Zustände gelöst (Grenze n₁ + n₂ ≤ N aus dem Schwanz von M/M/c); die Verzögerung folgt nach Little aus E[nₖ]/λₖ − 1.
- **Simulation** (`prio_simulation.py`): Ereignisse für Ankunft und Abfertigungsende, zwei Schlangen, bei unterbrechender Vorfahrt verdrängt ein Eiliger den zuletzt gestarteten Standard-Lkw, der mit seiner Restdauer
  an den Kopf der Standard-Schlange zurückkehrt; SplitMix64 mit getrennten Strömen (Zwischenankunft, Dauer, Klassenwahl); die ersten 5 % der Lkw werden nicht ausgewertet, alle werden zu Ende bedient.
- **Gegenproben:** (1) von Hand verfolgte Mini-Instanz mit vier Lkw für alle drei Reihenfolgen (Verzögerungen 2.3 / 2.567 / 2.5 nicht unterbrechend, 3.3 / 2.233 / 2.5 FIFO, 0 / 2.9 / 2.175 unterbrechend); (2) die Verdrängung
  (wer, welche Restdauer) von Hand; (3) die **Markov-Kette bestätigt die Formeln** (auch die Herleitung für mehrere Spuren) und die **Randverteilung der Gesamtzahl ist die von M/M/c**; (4) Simulation gegen Formeln und Kette;
  (5) ein von Hand gelöster Dreizustands-Fall der Kette; (6) Erhaltungssatz in Formel und Simulation.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund zweieinhalb Minuten parallel): 2 Spurzahlen × 3 Auslastungen (50, 80, 90 %) × 3 Reihenfolgen × 3 Streuungen, 20 % Eilige, je 6 Läufe à 1 000 000 Lkw;
  dazu eine Studie über den Anteil Eiliger (10 / 20 / 50 %, eine Spur, 90 %, exponentiell).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Verzögerungen in Abfertigungsdauern, wo nicht Minuten dasteht.

| Frage | Befund |
|---|---|
| Wie viel gewinnen die Eiligen, wie viel zahlen die anderen? | Eine Spur, 20 % Eilige, **jede** Dauer: nicht unterbrechend sind die Eiligen **(1 − pρ)/(1 − ρ) = 1.8 / 4.2 / 8.2-fach** schneller als bei FIFO, die Standard-Lkw **pρ/(1 − pρ) = 11.1 / 19.0 / 22.0 %** langsamer (Auslastung 50 / 80 / 90 %). Beides hängt nicht von cs² ab. Unterbrechend: **9.0 / 21.0 / 41.0-fach** schneller. |
| Stimmt die Simulation? | In allen 69 Zellen und Klassen mit Formel höchstens **2.3 %** Abweichung, im Mittel 0.5 %. Gemessen gegen den FIFO-Lauf derselben Zelle (exponentiell, eine Spur): nicht unterbrechend 1.80 / 4.20 / 8.23-fach, Standard +11.7 / +19.2 / +20.5 %; unterbrechend 8.93 / 20.98 / 41.34-fach, +22.9 / +24.1 / +26.0 %. |
| Vier Spuren? | Nicht unterbrechend 1.79 / 4.20 / 8.32-fach schneller, Standard +11.4 / +19.7 / +19.5 %; **unterbrechend warten die Eiligen weniger als 0.01 min** (bei allen drei Auslastungen). |
| Gilt der Erhaltungssatz? | **Ja für nicht unterbrechende Vorfahrt:** das Gesamtmittel ist nach Formel gleich FIFO (alle Zellen mit Formel); in der Simulation weichen die 18 Zellen höchstens 3.0 % ab (Rauschen). Vorfahrt **verschiebt**, sie spart nichts. |
| Und bei unterbrechender Vorfahrt? | Nur bei exponentieller Dauer (Zahl im System unabhängig von der Reihenfolge). Eine Spur, nach Formel gegenüber FIFO, Auslastung 50 / 80 / 90 %: bei cs² = 0 **+8.9 / +3.8 / +2.0 %**, bei cs² = 1 genau 0, bei cs² = 4 **−5.3 / −2.3 / −1.2 %**: bei streuender Dauer spart Unterbrechen etwas, weil lange Aufträge kurze nicht mehr blockieren. |
| Wie viele Eilige verträgt die Vorfahrt? | Eine Spur, 90 %, exponentiell, nicht unterbrechend, Anteil Eiliger 10 / 20 / 50 %: Gewinn der Eiligen **9.1 / 8.2 / 5.5-fach**, Verlust der Standard-Lkw **+9.9 / +22.0 / +81.8 %** (Formel). Simulierte Verzögerung der Eiligen 0.99 / 1.10 / 1.64 (FIFO 9.0), der Standard-Lkw 9.99 / 10.93 / 16.61 Abfertigungsdauern. |
| Wie gut ist die Markov-Kette? | Eine Spur, 90 %, 20 % Eilige, exponentiell, unterbrechend: Kette **0.21951 / 11.19511** gegen Formel 0.21951 / 11.19512, **12 880 Zustände**, Rest-Fehler unter 10⁻¹²; nicht unterbrechend Kette 1.09756 / 10.97560 gegen Cobham 1.09756 / 10.97561 (Abweichung unter 2·10⁻⁵, vom Abschneiden am Rand). |
| Wie verlässlich sind die Zahlen? | Der größte relative Standardfehler einer Klassenverzögerung (über 0.05 Abfertigungsdauern) beträgt 2.4 % (Mittel aus 6 Läufen). |
| Standardlauf der App? | Eine Spur, 90 %, 20 % Eilige, nicht unterbrechend, 150 000 Lkw, Seed 35: Eilige 3.33 min (Formel 3.29), Standard-Lkw **35.99 min (Formel 32.93, +9 %)**: für die Standard-Lkw deutlich zu hoch (+33 % statt +22 % gegenüber FIFO). |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Geschlossene Formeln statt Tabellen.** Die Vorab-Messreihe zeigte Gewinn und Verlust nur als Zahlen (8.2-fach, +22 % bei 90 %). Gerechnet ergibt sich, dass sie bei einer Spur **nicht von der Streuung der Dauer abhängen** und als
  (1 − pρ)/(1 − ρ) bzw. pρ/(1 − pρ) geschlossen angebbar sind; die Studie bestätigt das für cs² = 0, 1 und 4.
- **Der Vergleich „Unterbrechen spart bei streuender Dauer“ ist nur gegen die Formel sauber.** Die Simulation einer Zelle (cs² = 4, Auslastung 80 %) zeigt −5.8 % gegenüber dem FIFO-Lauf derselben Zelle, die Formel nur −2.3 %: der
  FIFO-Lauf liegt dort selbst 2 % über seiner Formel (schwerer Schwanz). Die README nennt deshalb die Formelwerte.
- **Vier Spuren, unterbrechend:** Der Gewinn der Eiligen in „Faktor“ (405- bis 953-fach) ist wegen Wartezeiten nahe null bedeutungslos; die App und diese README nennen dafür die absolute Wartezeit (unter 0.01 min).
- **Der Standardlauf der App liegt für die Standard-Lkw 9 % über der Formel** (150 000 Lkw bei 90 % Auslastung), weil bei hoher Last die Verzögerung der großen Klasse stark streut; die Studie mittelt sechs Läufe à 1 000 000.

## Ehrliche Grenzen

- Beide Klassen haben **dieselbe Abfertigungsdauer**. Bei kürzeren Aufträgen der Eiligen ändert sich W₀ = Σ λᵢ·E[Sᵢ²]/2; die Formel von Cobham gilt weiter, ist aber nicht gerechnet.
- **Zwei Klassen.** Die Formeln gelten für beliebig viele (σₖ kumuliert), die Demo zeigt zwei.
- Die Kette gilt nur für **exponentielle Dauer**; bei anderer Streuung ist (n₁, n₂) kein Markov-Zustand. Für nicht unterbrechende Vorfahrt gibt es nur bei **einer Spur** eine zweidimensionale Kette (bei vier Spuren müsste man
  die Klassen der Belegten mitführen, nicht gerechnet); die App zeigt dann die Kette der unterbrechenden Vorfahrt und sagt es.
- Mehrere Spuren: Formeln nur bei exponentieller Dauer (bei anderer Dauer nur Simulation).
- Wartekosten sind für beide Klassen gleich; bei unterschiedlichen Kosten ist die beste Rangfolge die nach Kosten je Dauer (cμ-Regel), nicht gerechnet.
- Unendlicher Warteraum, keine Abwanderung; bei Verlusten trifft es vor allem die untere Klasse (Stück 4 und 8).
- Die Studie hat 20 % Eilige (Anteil-Studie: 10 / 20 / 50 %) und die Auslastungen 50, 80, 90 %; die App zeigt für andere Werte die nächste Zelle und sagt es. Der Live-Lauf ist kurz und streut bei hoher Auslastung um Zehntel des Werts.

## Verwandte Demos im Portfolio

- [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) (Zusatzstück: die Grundlagen der Kette, die hier zweidimensional wird).
- [`mg1-kingman-demo`](https://github.com/sebastian-hanisch/mg1-kingman-demo) (Stück 10): eine Klasse, Pollaczek-Khinchine; hier mit zwei Klassen.
- [`mmc-queue-demo`](https://github.com/sebastian-hanisch/mmc-queue-demo) (Stück 3): mehrere Spuren, Erlang C.
- [`mm1-queue-demo`](https://github.com/sebastian-hanisch/mm1-queue-demo) (Stück 1): die Geburts-Sterbe-Kette einer Klasse.
- [`power-of-d-demo`](https://github.com/sebastian-hanisch/power-of-d-demo) (Stück 7): ein weiteres Gate, dessen mehrdimensionale Kette (2 und 3 Spuren) in den Tests als Referenz dient.
- [`truck-appointment-demo`](https://github.com/sebastian-hanisch/truck-appointment-demo): Terminvergabe für Lkw am Hafen.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Geduldige Lkw, unbegrenzter Warteraum | [Erlang A](https://github.com/sebastian-hanisch/erlang-a-demo), [Erlang B](https://github.com/sebastian-hanisch/erlang-b-demo) |
| Konstante Ankunftsrate | [Zeitvariable Ankünfte](https://github.com/sebastian-hanisch/time-varying-arrivals-demo) |
| Ein Gate | [Jackson-Netze](https://github.com/sebastian-hanisch/jackson-network-demo) |

Kein Folgestück: unterschiedliche Dauern und Wartekosten je Klasse, mehr als zwei Klassen.

## Tests

139 Tests, rund 45 Sekunden: Cobham und Preemptive-Resume von Hand, die geschlossenen Formeln für Gewinn und Verlust, der Erhaltungssatz, Sonderfälle (exponentielle Dauer) und Fälle ohne Formel, die Markov-Kette
(von Hand gelöster Dreizustands-Fall, Gesamtzahl gleich M/M/c, Übereinstimmung mit den Formeln, Güte der Lösung), die Simulation (Mini-Instanz für alle drei Reihenfolgen, Verdrängung von Hand, Invarianten), Simulation gegen
Formeln und gegen die Kette, Vollständigkeit der vorgerechneten Datei, Presets und Permalink, Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen
Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `prio_formulas.py` | Cobham, Preemptive-Resume, Erhaltungssatz, geschlossene Formeln für Gewinn und Verlust |
| `prio_chain.py` | exakte Markov-Ketten (unterbrechend beliebig viele Spuren, nicht unterbrechend eine Spur) |
| `prio_simulation.py` | Ereignisse, Vorfahrt, Verdrängung |
| `prio_evaluation.py` | Live-Lauf, Studienzellen, Auswahl der Kette |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `prio_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `prio_presets.py`, `prio_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Cobham, A. (1954): Priority assignment in waiting line problems. *Operations Research* 2(1), 70–76 (Verzögerung je Klasse für eine Spur mit beliebiger Dauer und für mehrere Spuren mit exponentieller Dauer, nicht unterbrechend).
- Erhaltungssatz (Kleinrock) und die Formel für unterbrechende Vorfahrt mit Wiederaufnahme sind Standardergebnisse der Warteschlangentheorie; Quellen nicht einzeln belegt, die Formeln sind hier durch die Markov-Kette unabhängig geprüft.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py`.

Gebaut mit Streamlit und Plotly (die Markov-Kette löst scipy).
