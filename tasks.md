
# tasks.md — Qwen3‑VL → Qwen‑Image Identity/State Injection (Slot‑basiert)

Ziel: In kurzer Zeit ein **lauf- und testfähiges System** aufbauen, das
- **1–n Referenzbilder** je Identität/Zustand nutzt,
- Identität **slot‑basiert** in **beliebige Prompts** injiziert,
- **Qwen3‑VL** (Analyse, eingefroren) + **Qwen‑Image** (Generierung, Diffusers) korrekt verwendet,
- **Gerät = CUDA** und **dtype** (fp16/bf16 oder fp32) **konsistent** handhabt,
- ohne Diffusers‑Spezialwissen verständlich ist.

Die Schritte bauen aufeinander auf: erst ein minimaler Render‑Pfad, dann Inversion („Teacher“), dann amortisierte Regression und schließlich Multi‑ID/Produktionshärtung. **Kein Code** – nur eindeutige, umsetzbare Tasks.

---

## Phase 0 — Umgebung & Grundspezifikation (verbindlich)

1. **GPU‑Verfügbarkeit & Abbruchbedingung**
   - Stelle sicher, dass eine NVIDIA‑GPU mit CUDA verfügbar ist. Falls `torch.cuda.is_available()` *false* liefert, **sofort mit klarer Fehlermeldung abbrechen**. CPU wird **nicht** unterstützt.
   - Dokumentiere, welche CUDA‑Version, Treiber und GPU‑Speicher verfügbar sind (für spätere VRAM‑Abschätzungen).

2. **Geräte- und dtype‑Vorgaben (global)**
   - **Gerät:** immer `"cuda"`.
   - **dtype:** wähle **fp16** (oder **bf16**, wenn GPU das stabil unterstützt). Nur wenn notwendig, auf **fp32** zurückfallen.
   - **Konsequenz:** Alle Tensoren (Text‑Embeddings, Residuals, Latents) und alle Modul‑Gewichte werden **explizit** auf **dasselbe** Gerät und **denselben** dtype gebracht, bevor sie miteinander interagieren.
   - **Prüfung:** Vor jeder Vorwärtsrechnung die Geräte/dtypes der beteiligten Tensoren vergleichen und bei Abweichung **hart** abbrechen (mit Fehlermeldung, die den Namen/Shape/dtype des fehlerhaften Tensors nennt).

3. **Datenlayout (verbindlich)**
   - Pro Identität/Zustand existiert ein Ordner `.../seed_<id>/` mit **mindestens 2 Bildern** (empfohlen: 2–5).
   - Bildformate: PNG/JPG/WEBP/BMP. Dateinamen beliebig.
   - **Akzeptanzkriterium:** Ein Verzeichnis‑Scanner erzeugt eine Liste aller `seed_*`‑Ordner mit ≥2 Bildern.

4. **Platzhalter‑Konvention (einheitlich)**
   - Verwende **einen** festen Platzhalter‑String in allen Prompts: z. B. `~ID~`.
   - **Wichtig:** Dieser String wird **niemals** verändert (keine Suffixe wie `~ID42~`). Die Unterscheidung der Personen geschieht **außerhalb** des Textes durch die zugeordneten Residual‑Embeddings.
   - **Folge:** Die Tokenisierungslänge des Platzhalters ist **konstant**; alle Slot‑Residuals haben dieselbe Sequenzlänge **L**. Das verhindert Längenmismatches.
   - **Multi-Person-Zuordnung (Variante A):** Bei mehreren Slot-Vorkommen im Prompt (z.B. `~ID~ and ~ID~ on a beach`):
     * Slots werden in **Reihenfolge des Auftretens** (links→rechts) den Personen zugeordnet
     * `slots = [(s_0, L), (s_1, L), ...]` → `residuals = [R_person0, R_person1, ...]`
     * Injection: `E_base[:, s_k:s_k+L, :] += α_k · R_k` für jede Person k

5. **Tokenizer‑Einstellung (konstant)**
   - Texte **immer** mit `padding="max_length"` und `truncation=True` tokenisieren.
   - **Akzeptanzkriterium:** Die Embedding‑Sequenzlänge **S** ist bei **allen** Prompts identisch (z. B. 256/512, je nach Modell).

---

## Phase 1 — Minimaler Render‑Pfad (ohne Training) **[MVP in 1–2 Stunden]**

Ziel: Prompt‑Embeddings direkt an Qwen‑Image übergeben und Bilder erzeugen. Dient als Rauchtest für Gerät, dtype, Shapes und Diffusers‑API.

1. **Qwen‑Image Pipeline laden**
   - Lade die Diffusers‑Pipeline für Qwen‑Image.
   - Stelle sicher, dass folgende Komponenten **exponiert** sind: `tokenizer`, `text_encoder`, `unet`, `vae`, `scheduler`, `image_processor`.
   - **Prüfe** die Aufrufsignatur der Pipeline: Bildgenerierung mit **`prompt_embeds`** und optional **`negative_prompt_embeds`** muss funktionieren (d. h. keine Nutzung von String‑Prompts in diesem MVP).

2. **Base‑Prompt encodieren**
   - Wähle einen **neutralen Base‑Prompt mit Slot**, z. B. `a portrait photo of ~ID~, studio lighting`.
   - Tokenisiere und encodiere zu `E_base` mit Shape **[1, S, H]** (S = Sequenzlänge, H = Embedding‑Dim).
   - **Akzeptanzkriterium:** `E_base.device == "cuda"` und `E_base.dtype == global_dtype`.

3. **Leere Injection testen**
   - Erzeuge Dummy‑Residual `R_slot = 0` mit Shape **[1, L, H]** (L = Tokenanzahl des Platzhalters). Injektion: Kopiere `E_base` in `E_inj`; **ohne** Änderung.
   - Generiere 1–2 Bilder mit festen Seeds. **Erwartung:** Korrekte Outputs, keine Fehler, reproduzierbar zwischen Seeds.

4. **Alpha‑Sweep‑Gerüst**
   - Definiere eine Liste `alphas` (z. B. 0.5, 1.0, 1.5, 2.0), `seeds` (z. B. 0–7).
   - Erzeuge einen **Grid‑Renderer** (Zeilen = Alphas, Spalten = Seeds). Speichere PNG + JSON‑Metadaten (Alphas, Seeds, Schritte, Guidance, Bildgröße).

---

## Phase 2 — Teacher‑Inversion (slot‑rein) **[Testbar in < 1 Tag]**

Guter Punkt — hier ist **Phase 2 (Teacher-Inversion) vollständig und eindeutig** beschrieben: *was* trainiert wird, *wie* der Forward/Backward-Pass aussieht, *welche* Losses & Metriken genutzt werden, und *wann* du speicherst bzw. abbrichst. (Nur Prosa, kein Code.)

---

# Phase 2 — Teacher-Inversion (slot-rein, mit Alpha-QA)

## Ziel

Für eine zu konfigurierende Liste an Identitäten (Ordner `seed_<id>`) wird je ein **teacher_prompt_embeds.pt** erzeugt, das

* exakt zur **Tokenizer-Länge S** und **Embedding-Dim H** deiner Qwen-Image-Textencoder-Embeddings passt (Form `[1, S, H]`),
* **nur im Slot-Fenster** (Tokenbereich des Platzhalters `~ID~`) eine nicht-null Residual-Änderung enthält,
* bei Alpha-Sweeps ((\alpha) 0.5–2.0) die **Identität sichtbar** macht, ohne Stil/Setting des Prompts zu zerstören.

---

## Voraussetzungen (fix)

* **Gerät:** immer `cuda`. Vor Start sicherstellen, dass alle Module/Tensoren auf CUDA liegen.
* **dtype:** global (`fp16` oder `bf16`, nur falls nötig `fp32`).
* **Pipeline:** Qwen-Image als Diffusers-Pipeline mit Zugriff auf `tokenizer`, `text_encoder`, `unet`, `vae`, `scheduler`, `image_processor`.
* **Base-Prompt mit Slot:** z. B. `a portrait photo of ~ID~, studio lighting`. Dieser **eine** Base-Prompt wird für alle Identitäten verwendet.
* **Tokenizer-Einstellung:** `padding="max_length"`, `truncation=True`; `max_length = tokenizer.model_max_length`. So ist **S konstant**.
* **Slot-Fenster:** Bestimme exakt einmal die Token-Position des Platzhalters in `E_base` → **Startindex `s`**, **Länge `L`** (konstant für alle Identities).

---

## Daten je Identität

* 2–5 RGB-Bilder (idealerweise 3), in `seed_<id>/`.
* Vorverarbeitung: zentrierter Crop (falls nötig), dann **Resize auf die geplante Renderauflösung** (z. B. 768×512). Keine aggressive Augmentation; optional leichte horizontale Flips, falls identitätsverträglich.

---

## Zu lernende Parameter (pro Identität)

* **P**: ein lernbares Tensor-Feld ([1, S, H]), initial 0.
* **log_scale**: ein skalierter, lernbarer Skalar (Start 0) → `scale = exp(log_scale)`; **Clamp** auf einen Bereich (empfohlen [0.01, 1.5]).
  *Begründung:* Stabilisiert Normen (verhindert RMS-Kollaps/Explosion in fp16/bf16).

> Wir lernen **kein** neues Netz, sondern nur diese Embedding-Abweichung.

---

## Re-Parametrisierung (Pflicht)

* **Normierung:** (\hat{P} = P / \mathrm{RMS}(P)) (RMS über alle Token-Kanäle; (\epsilon)-abgesichert).
* **Residual:** (R = \text{scale} \cdot \hat{P}).
* **Aktuelles Embedding:** (E = E_\text{base} + R).
* **Gerät/dtype-Konsistenz:** `E_base`, `R`, `E` müssen **auf CUDA** und im **globalen dtype** liegen.

---

## Forward-Pass (pro Optimierungsschritt)

1. **Latents der Zielbilder:** Einmalig zu Beginn pro Identität via VAE in Latents umwandeln und **cachen** (Zeit sparen).
2. **Minibatch aus Bildern:** Ziehe `batch_size` Bilder aus dem Cache (z. B. 2).
3. **Timestep & Noise:** Ziehe **gleichverteilte** Timesteps (t) aus `[0, num_train_timesteps)`, und ziehe weißes Rauschen (\epsilon) mit gleicher Form wie die Latents.
4. **Add Noise:** Rechne (z_t = \mathrm{add_noise}(z_0, \epsilon, t)) mit dem **Scheduler** der Pipeline.
5. **E skalieren:** Nutze (E) wie oben; **optional CFG**: Erzeuge analog `E_neg` aus einem **fixen Negativ-Prompt** (`" "` - Leerstring für alle Identitäten).
6. **UNet-Vorhersage:**

   * **Ohne CFG:** (\hat{\epsilon} = \mathrm{UNet}(z_t, t, \text{encoder_hidden_states}=E)).
   * **Mit CFG:** Standard-Kombination ( \hat{\epsilon} = \hat{\epsilon}*\text{uncond} + \text{cfg} \cdot (\hat{\epsilon}*\text{cond} - \hat{\epsilon}_\text{uncond})) mit **identischen Shapes** für cond/uncond.
7. **Zielgröße:**

   * Wenn `scheduler.config.prediction_type == "v_prediction"`: Ziel ist die **Geschwindigkeit** (v) (Scheduler-Funktion verwenden).
   * Sonst Ziel = (\epsilon).
     *Beides ist in Diffusers standardisiert — wichtig ist, genau die zum Scheduler passende Zielgröße zu wählen.*

---

## Loss-Funktion (pro Schritt)

Setze die gewichtete Summe:

* **Noise-Prediction-MSE (Pflicht):**
  (\mathcal{L}*\text{mse} = |\hat{\epsilon} - \epsilon*\text{target}|*2^2) *oder* (|\hat{v} - v*\text{target}|_2^2) (je nach Scheduler).
* **RMS-Regularisierung (empfohlen, klein gewichtet):**
  Halte die pro-Token-RMS des **Slot-Segments** von (E - E_\text{base}) in einem sinnvollen Bereich (z. B. 0.6–1.4 des Trainingsmittelwerts). Praktisch: L1 auf die Differenz der RMS zum laufenden Mittelwert.
* **Off-slot-Sparsity (Pflicht, falls Maskierung *am Ende* erfolgt):**
  Strafterm, der ((E - E_\text{base})) **außerhalb** des Slot-Fensters Richtung 0 zieht (z. B. L2 mit kleinem Gewicht).
  *Hinweis:* Wenn du **bei jedem Schritt** die Off-slot-Anteile hart auf 0 setzt, entfällt dieser Term.

**Empfohlene Gewichte (Startwerte):**
(\mathcal{L} = 1.0\cdot \mathcal{L}*\text{mse} + 0.02\cdot \mathcal{L}*\text{rms} + 0.05\cdot \mathcal{L}_\text{offslot})

---

## Optimierung

* **Optimizer:** AdamW auf ({P, \text{log_scale}}).
* **Learning Rate:** 5e-3 (Start), bei Divergenz 2e-3.
* **Schritte pro Identität:** 400–800 (Startwert 600).
* **CFG (optional):** 4.0–6.0; bei Instabilität auf 3.0–4.0 gehen.
* **Batchgröße:** 2 (bei 2–5 Bildern meist ausreichend).
* **Gradient-Clipping:** globaler Clip auf **1.0** (fp16 empfiehlt sich).
* **Checkpointing:** optional alle 200 Schritte Zwischenergebnis schreiben.

---

## Metriken (während des Trainings)

* **Train-Loss (MSE/v-MSE):** muss über die Schritte **fallen** (glatt oder mit kleinen Schwankungen).
* **RMS-Monitor:**

  * Gesamt-RMS von (R = E - E_\text{base})
  * **Slot-RMS** (nur ([s:s+L)))
  * **Off-slot-RMS** (außerhalb des Slots, sollte → 0 gehen)
* **Scale-Wert:** `exp(log_scale)` sollte in einem sinnvollen Intervall landen (typ. 0.2–1.2). Wenn dauerhaft nahe 0.01 → zu aggressiver Off-slot- oder RMS-Term; bei 1.5 „deckelt“ die Clamp → ggf. LR/CFG prüfen.

---

## Stopping-Kriterien

* **Regulär:** Erreiche 600 Schritte *oder* die Train-Loss sinkt in den letzten 100 Schritten < 1 % (Plateau).
* **Frühabbruch:**

  * **Numerik-Warnzeichen:** NaNs/Inf in Loss/Scale; sofort stoppen, LR senken, CFG senken, ggf. fp32 testen.
  * **Divergenz:** Loss steigt 50 Schritte am Stück → LR halbieren, weitermachen, sonst abbrechen.

---

## Post-Processing (Slot-Maskierung, Pflicht)

* Berechne ( \Delta = E - E_\text{base} ).
* **Setze außerhalb des Slot-Fensters ([s:s+L))** alle Werte **exakt auf 0** (harte Maskierung).
* Erzeuge (E_\text{teacher} = E_\text{base} + \Delta_\text{masked}).
* **Speichere** `teacher_prompt_embeds.pt` (Form `[1, S, H]`, **auf CPU speichern**, Metadaten notieren: S, H, L, s, dtype, scheduler, cfg, schritte).
* **Portabilität:** Das slot-maskierte Residual ist direkt in beliebige Inferenz-Prompts injizierbar, da die Maskierung die Änderung auf den Slot-Bereich beschränkt. Kein "Re-Basing" notwendig.

---

## Alpha-QA (direkt nach dem Speichern)

* **Alphas:** 0.5, 1.0, 1.5, 2.0 (bei schwachem Effekt: bis 3.0 erweitern).
* **Seeds:** 4–16 fixe Seeds.
* **Generierung:**

  * `E_eval(α) = E_base + α · (E_teacher − E_base)`
  * Render mit denselben Pipeline-Parametern (Schritte, CFG, Auflösung) wie geplant.
* **Erwartung:** Mit steigendem α wird die Identität **klarer**; Stil/Setting des Base-Prompts bleibt.
* **Artefakt-Check:** Übersteuerung → α senken; „kein Effekt“ → LR/Schritte/CFG prüfen (und Scale-Wert).
* **Ablage:** Grid + JSON (Alphas, Seeds, Schritte, Guidance, H/W, dtype) in `seed_<id>/alpha_eval/`.

---

## Akzeptanzkriterien (pro Identität)

* **Technisch:**

  * `teacher_prompt_embeds.pt` existiert, Shape **[1, S, H]** korrekt, dtype dokumentiert.
  * Off-slot-RMS < **1e-6** (oder identisches Null-Array), Slot-RMS im sinnvollen Bereich (z. B. 0.1–2.0 je nach Modell).
  * Keine dtype/device-Mischungen; CFG-Pfad formkompatibel (cond/uncond gleiche S/H).
* **Visuell:**

  * Bei mind. **einem** α ∈ [1.0, 1.8] ist Identität **klar erkennbar**, ohne dass Hintergrund/Setting „umkippt“.
  * Wiederholbarkeit über Seeds (Identität soll persistieren, Varianz im Bildaufbau ist ok).


---

## Phase 3 — Amortisierte Regression (Qwen3‑VL → Slot‑Residual) **[MVP in 1–2 Tagen]**

Ziel: Einen **Head** trainieren, der aus 1–n Referenzbildern ohne Inversion direkt das **Slot‑Residual** vorhersagt.

1. **Qwen3‑VL Feature‑Extraktion (frozen)**
   - Lade Qwen3‑VL (Gerät = CUDA, dtype = global).
   - **Pixel‑Budget**: Für 1–3 Bilder pro Identität ein **Gesamt‑Budget** festlegen (z. B. 2.0 MP); skaliere alle Bilder proportional herunter; begrenze die **maximale Kantenlänge** (z. B. 1536).
   - Extrahiere **visuelle Tokens** des Vision‑Towers; bilde ein **gepooltes Feature** (z. B. Mean über Token und über Bilder). **L2‑normalisiere** das Ergebnis → Vektor `z` (Shape `[D]`).
   - **Akzeptanzkriterium:** `‖z‖₂ ≈ 1.0` (numerisch geprüft).

2. **Ziel‑Residual vorbereiten**
   - Lade `E_teacher` aus Phase 2 (gleicher Base‑Prompt!).
   - Definiere `R_teacher_slot = (E_teacher - E_base)[s:s+L, :]` (nur Slot‑Segmente; Shape **[L, H]** pro Beispiel; im Batch **[B, L, H]**).

3. **Head‑Spezifikation (Linear‑Baseline)**
   - Head‑Input: `z`.
   - Head‑Output: **[L, H]** pro Beispiel; im Batch **[B, L, H]** (exakt dieselbe Slot‑Länge **L** und Embedding‑Dim **H** wie beim Text‑Encoder).
   - **Injection-Dimensionen (explizit):**
     * Arbeite immer mit ge-batchten Prompt-Embeddings: `E_base ∈ ℝ^{B×S×H}` (bei Einzelinferenz B=1)
     * Residual: `R ∈ ℝ^{B×L×H}`
     * Injection pro Batch-Eintrag b: `E_base[b, s:s+L, :] += α_b · R[b, :, :]`
     * **Kein implizites Broadcasting**; Dimensionen explizit matchen
     * Einzelfall (B=1): Wenn Head intern `R ∈ ℝ^{L×H}` liefert, vor Injection zu `R[None, ...] ∈ ℝ^{1×L×H}` erweitern
   - **Re‑Parametrisierung:** Per‑Token RMS‑Normierung des Outputs + **Skalen‑Clamp** (exp(log_scale) ∈ [0.01, 1.5]).
   - **Gerät/dtype:** Head auf CUDA und globalem dtype; Input `z` auf dasselbe Gerät/dtype casten.

4. **Loss & Training**
   - **Loss:** `0.7 * (1 − CosSim)` + `0.3 * MSE` + `0.05 * |RMS(pred) − RMS(target)|`.
   - **Batching:** Pro Schritt mehrere Identitäten mischen.
   - **Eval:** Halte ein Validierungsset zurück; tracke CosSim und RMS‑Fehler.
   - **Optional (selten, stark):** Alle N Schritte ein **Low‑Res‑Render** (256px, wenige Diffusionsschritte) mit injiziertem Slot‑Residual aus dem Head; miss Bild-Ähnlichkeit mit **ArcFace** als Metrik (nur "functional grounding", **nicht** im Loss).

5. **Checkpoints & Reproduzierbarkeit**
   - Speichere Head‑Gewichte regelmäßig (nach Epochen).
   - Fixiere Seeds (global) und dokumentiere `alphas`, Bildgrößen, CFG, Schrittzahlen.

---

## Phase 4 — Inferenz‑Pipeline (1–n Personen, beliebige Prompts) **[Produktions‑MVP]**

1. **Prompt mit Slots schreiben**
   - Beispiel: `boy riding a bike with ~ID~ at sunset` (für eine Person) oder `~ID~ and ~ID~ on a beach` (für zwei Personen).
   - **Wichtig:** Immer **derselbe** Platzhalter‑String `~ID~` (mehrere Vorkommen erlaubt).

2. **Slot‑Suche (Token‑basiert, deterministisch)**
   - Tokenisiere den Prompt (padding=max_length, truncation=True). Suche die Token‑Sequenz des Platzhalters; erhalte Startindex **s_k** und Länge **L** (konstant).
   - **Fehlerfall:** Wird kein Slot gefunden, **Abbruch** mit Fehlermeldung (Prompt muss `~ID~` enthalten).

3. **Base‑Embeddings**
   - Encodiere den **vollen Prompt** zu `E_base` **[1, S, H]** auf CUDA/global‑dtype.
   - Prüfe, dass S und H den in Phase 1/2 verwendeten Werten entsprechen.

4. **Residuals je Slot beschaffen**
   - **Qwen3-VL-basiert (einzige Variante):** Qwen3-VL → Feature `z` aus 1-n Referenzbildern der Zielperson; Head-Vorhersage `R_slot_pred` **[L, H]** (bei Einzelinferenz) oder **[B, L, H]** (im Batch).
   - **Prüfung:** `R_slot.device == E_base.device` und `R_slot.dtype == E_base.dtype`.
   - **Shape-Anpassung:** Falls `R_slot ∈ ℝ^{L×H}`, erweitere zu `R_slot[None, ...] ∈ ℝ^{1×L×H}` vor Injection.

5. **Injection mit Alpha**
   - Wähle **α_k** pro Person (empfohlen: Sweep 0.8-1.8; **unterschiedlich** je Person möglich, von außen als Parameter angegeben).
   - Für **jedes Vorkommen** des Slots im Prompt: `E_base[:, s_k:s_k+L, :] += α_k · R_k[:, :, :]`.
   - **Mehrere Personen:** Wiederhole den Schritt je Person; nutze getrennte Referenzbilder/Heads und α-Werte.
   - **Reihenfolge:** Slots werden in Auftretensreihenfolge (links→rechts) den Personen zugeordnet.

6. **Rendering**
   - Rufe Qwen‑Image mit **`prompt_embeds=E_base`** auf (optional `negative_prompt_embeds`); setze gewünschte **Seeds**, **Guidance‑Scale**, **Schritte**, **Auflösung**.
   - **Akzeptanzkriterium:** Outputs entstehen ohne Gerät/dtype/Shape‑Fehler und sind zwischen Seeds reproduzierbar.

7. **QA (Alpha‑Sweep)**
   - Für neue Personen/Prompts einmalig ein Grid (Alphas × Seeds) erzeugen und visual prüfen: **Identität↑** mit α↑, Szene/Action bleiben steuerbar.

---

## Phase 5 — Multi‑ID & robuste Kantenfälle

1. **Mehrere Slots im Prompt**
   - Slots **nicht überlappen lassen**. Bei Dichtlage mindestens 1 Token Abstand sicherstellen.
   - Die **längere** Slot‑Sequenz zuerst matchen (greedy), dann die kürzeren, um Verwechslungen zu vermeiden.
   - **Injection-Ablauf:**
     * `E_base ∈ ℝ^{1×S×H}` einmal encodieren
     * Für jede Person k: `R_k ∈ ℝ^{1×L×H}` holen (oder als Mini-Batch B=K berechnen)
     * Nacheinander in dieselbe Kopie von `E_base` injizieren: `E_base[:, s_k:s_k+L, :] += α_k · R_k[:, :, :]`
     * Reihenfolge der Injection egal (Addition ist kommutativ); L fix und identisch für alle Slots

2. **Konfliktauflösung**
   - **Prompt widerspricht Identität** (z. B. „boy“ vs. erwachsene Person): α **senken** und Prompt harmonisieren; sonst gewinnt häufig der Text.
   - **Übersteuerung** (zu hoher α): Artefakte oder Stilbruch → α reduzieren, ggf. `negative_prompt` schärfen.

3. **Längen‑Mismatches (sollten nicht auftreten)**
   - Weil der Platzhalter **immer gleich** ist, ist **L konstant**. Falls dennoch Unterschiede entstehen (Tokenizer‑Versionen), wähle **hart**: Pipeline stoppen und die Tokenizer‑Version angleichen; **kein** automatisches Pad/Trunk in Produktion.

---

## Phase 6 — Operationalisierung & Monitoring

1. **Artefakt‑Ablage (Standard)**
   - Pro Identität: `teacher_prompt_embeds.pt`, Alpha‑Grid, Meta‑JSON (Alphas, Seeds, CFG, Steps, Auflösung).
   - Für den Head: Checkpoints mit klarer Versionskennung (Datum, Commit‑Hash, Trainings‑Hyperparameter).

2. **Konfigurationsdatei (YAML/JSON)**
   - Enthält **alle** Konstanten: Gerät, dtype, S/H (vom Text‑Encoder), Slot‑String, L, Pixel‑Budget, max‑side, CFG‑Bereich, Standard‑Seeds, Standard‑Auflösungen.

3. **Monitoring‑KPIs**
   - **Offline:** CosSim(Head‑Pred, Teacher‑Slot) auf Val‑Set; RMS‑Fehler.
   - **Online:** Visuelle QA‑Passrate; Verteilung optimaler α; Anteil Fehlerfälle (Slot nicht gefunden; Gerät/dtype‑Mismatch).

4. **Determinismus**
   - Seeds und Library‑Determinismus (soweit möglich) dokumentieren; Abweichungen (z. B. cuDNN) festhalten.

5. **Ressourcen/Skalierung**
   - Für 20.000 Identitäten: Teacher‑Inversion **batchen/sharden** (mehrere Worker, identische Hyperparameter). Head‑Training verteilt (Data‑Parallel).

---

## Phase 7 — Abnahme‑kriterien (Definition of Done)

- **MVP:** Phase 1 + Phase 2 vollständig; pro Identität entsteht ein Grid mit klar sichtbarer Identität (≥1 α), ohne Gerät/dtype/Shape‑Fehler.
- **Amortisiert:** Phase 3 trainiert; Inferenz in Phase 4 erzeugt Bilder **ohne** vorgängige Inversion – nur aus Referenzbildern.
- **Multi‑ID:** Mindestens ein Prompt mit **zwei** Slots rendert beide Personen stabil.
- **Robustheit:** Pixel‑Budget verhindert VRAM‑Spitzen; Fehlerpfade geben klare Meldungen (Slot fehlend, dtype/Device falsch).

---

## Anhänge — Eindeutige technische Konstanten & Prüfungen

- **Shapes:**
  - `E_base`: **[B, S, H]**, `S` konstant (Tokenizer‑Padding), `H` = versteckte Dim des Text‑Encoders (bei Einzelinferenz B=1).
  - `R_teacher_slot` und `R_head_pred`: **[L, H]** pro Beispiel; im Batch **[B, L, H]**; `L` = Tokenanzahl des Platzhalters (konstant).
  - **Kein implizites Broadcasting bei Injection**; immer explizit auf **[B, L, H]** bringen vor Addition.
- **Gerät:** **immer** `"cuda"`. Vor jedem Rechenschritt prüfen, dass **alle** beteiligten Tensoren/Module auf `"cuda"` liegen.
- **dtype:** global (fp16/bf16 oder fp32). Keine stillen Casts zwischen dtypes; **hart** abbrechen, wenn abweichend.
- **Tokenizer:** immer `padding="max_length"`, `truncation=True`; identische Version/Checkpoint im gesamten System.
- **Slot‑String:** exakt `~ID~`. Nicht abwandeln, nicht parametrisieren. Mehrere Vorkommen sind erlaubt; **dieselbe** Tokenlänge **L** an allen Stellen.
