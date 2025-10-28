# 1) Ziel / Scope

**Ziel:** Aus 1–n Referenzbildern (pro Identity) wird ein **Residual-Embedding pro Platzhalter-Slot** gelernt, das in **beliebige Prompts** injiziert werden kann (z. B. *"boy riding a bike with ~ID42~ at sunset"*).
**Generator:** Qwen-Image (Diffusers, konsumiert `prompt_embeds`).
**Analyse:** Qwen3-VL (frozen) → visueller Feature-Vektor aus 1–n Bildern.

# 2) Artefakte / Daten

* **Rohdaten:** `/data/.../seed_<id>/{img_*.jpg,...}` (20k Identities).
* **Base-Prompt mit Slot:** z. B. `a portrait photo of ~ID~, studio lighting` (global fixiert für T0, konfigurierbar).
* **Teacher-Embeds:** `teacher_prompt_embeds.pt` ([1,S,H]) pro Identity, **für den Base-Prompt mit Slot**.
* **Slot-Maske:** Token-Start/-Länge des Platzhalters im Base-Prompt (Indexbereich [s, s+L)).
* **Slot-Config:** Fixe Token-Subsequenz T_slot=[t₁,...,t_L] und Länge L (default: 4), versioniert und eingeforen.

# 3) Slot-Definition & Tokenisierung

**Zentrale Regel:** Kein Tokenizer-Hack, keine Vokab-Erweiterung. Wir nutzen einen **fixen Platzhalter-String** (z.B. `~ID~`), wie er vom gepinnten Qwen-Image-Tokenizer zerlegt wird.

**Warum kein resize_token_embeddings()?**
* Vokab-Erweiterungen verändern Embedding-Tabellen → Inkompatibilität mit Checkpoints.
* Wir injizieren direkt in `prompt_embeds`, nicht über Token-Embeddings.

**Vorgehen zur Slot-Bestimmung:**

1. **Slot-String wählen:** Wähle einen Kandidaten-String (z.B. `~ID~`, `~identity~`, `~ID_token~`), der mit dem gepinnten Tokenizer **exakt L=4 Subtokens** ergibt.
2. **Tokenisierung (einmalig):**
   ```python
   tokens = tokenizer(
       slot_string,
       padding="max_length",
       truncation=True,
       max_length=tokenizer.model_max_length,
       add_special_tokens=False,
       return_tensors="pt"
   )
   T_slot = tokens.input_ids[0, :L].tolist()  # [t₁, t₂, t₃, t₄]
   ```
3. **Config einfrieren:** Speichere `slot_string`, `T_slot=[t₁,...,t_L]`, und `L` in Config/Meta (versioniert).
4. **Slot-Suche (Runtime):** Bei jedem Prompt: tokenisiere, suche `T_slot` als Subsequenz in `input_ids`, injiziere an Position `[s:s+L]`.

**Akzeptanzkriterium:**
* Slot wird deterministisch gefunden; wenn nicht: Abbruch mit Fehler „Slot ~ID~ nicht gefunden; Tokenizer/Version/Prompt inkonsistent".
* L ist fix (default: 4); Tokenizer-Checkpoint & Version dokumentiert & eingefroren.

# 4) Systemübersicht (Stages)

**T0 – Teacher-Erzeugung (einmalig / batchbar):**
Invertiere pro Identity die Qwen-Image-`prompt_embeds` gegen die vorhandenen Bilder (Noise-Prediction-Loss). Ergebnis: `teacher_prompt_embeds.pt`. **Maskiere** danach das Delta **außerhalb des Slot-Fensters auf 0**:
[
R^\text{teacher} ;=; \big(;E_\text{teacher} - E_\text{base};\big)\odot M_\text{slot}
]
(Maske (M_\text{slot}) = 1 im Slot, 0 sonst). → **Slot-reines Residual**.

**T1 – Amortisierte Regression (einmalig trainieren):**
Qwen3-VL (frozen) extrahiert Feature (z) aus 1–n Bildern; ein **Head** (Linear-Baseline, optional MLP) lernt
[
\hat R ;=; f_\theta(z) ;\approx; R^\text{teacher}_{[s:s+L,:]}
]
also **nur das Slot-Residual** (Form ([L,H])). **Reparam**: per-token RMS-Norm + **Skalen-Clamp**.

**INF – Inferenz (laufend):**
Für jeden Prompt (P) mit **1–n Slots** `~IDk~`:

* `E_base = encode(P)`
* Für jede Person (k): Slotsuche → Segment ([s_k:s_k+L_k)); `R_k` via Head (aus 1–n Bildern)
* **Injection:** `E_base[:, s_k:s_k+L_k, :] += α_k · R_k`
* → `prompt_embeds = E_base` an Qwen-Image → Bild.

# 5) Funktionale Anforderungen (Tests – Given/When/Then)

### T0 – Teacher & Maskierung

* **Given** 2–5 Bilder pro Identity, Base-Prompt `a portrait of ~ID~`.
* **When** Inversion läuft 400–800 Steps, CFG=4–6.
* **Then** entsteht `teacher_prompt_embeds.pt` s.t. ( |R^\text{teacher}| > 0) **nur im Slot**; außerhalb < (10^{-6}).
* **Then** Alpha-Sweep-Grid zeigt mit (α∈{0.5,1.0,1.5,2.0}) **zunehmend stabile Identität** bei konstantem Stil.

### T1 – Regressor

* **Given** (z) aus Qwen3-VL (Pixel-Budget ≤2 MP über 1–3 Bilder).
* **When** Head (Linear) wird auf (R^\text{teacher}_{slot}) trainiert (Loss = (0.7·)cos + (0.3·)MSE, +RMS-Alignment).
* **Then** Val-Cos-Sim (\ge 0.85) gegen Teacher-Slot-Residual; RMS(\in[μ\pm 2σ]) der Teacher-RMS-Verteilung.

### INF – 1–n Personen / beliebige Prompts

* **Given** Prompt `boy riding a bike with ~ID42~ and ~ID7~ at sunset`.
* **When** Slot-Fenster korrekt gefunden, `α∈[0.8,2.0]`, Injection nur in die jeweiligen Fenster.
* **Then** gerenderte Bilder enthalten **beide** Identitäten konsistent; Szene/Textanteile bleiben steuerbar (keine Überschreibung außerhalb der Slots).

# 6) Nicht-funktionale Anforderungen

* **Zeit/VRAM:** Pixel-Budgetierung (max-side, Gesamt-MP ≤2.0 MP) bei Qwen3-VL; fp16/bf16 wo stabil.
* **Skalierung:** 20k Identities → T0 & T1 sind **batchbar**; Checkpoints + Sharding der Teacher-Dateien.
* **Robustheit:** Slot-Maskierung verhindert Prompt-Drift; Residual-Clamp vermeidet RMS-Kollaps.
* **Tokenizer-Versionierung:** Qwen-Image-Tokenizer-Checkpoint + Version eingefroren (in requirements/config dokumentiert); kein resize_token_embeddings().

# 7) Training im Detail

### T0 – Teacher (per Identity)

**Ziel:** Eigenes, leichtes Inversions-Training (Noise-Prediction-Loss) – **nicht** die generische Textual-Inversion aus diffusers.

1. **Encode:** `E_base = encode(BASE_PROMPT)` mit konfiguriertem Base-Prompt (z.B. `"a portrait photo of ~ID~, studio lighting"`).
2. **Optimieren:** Lerne P s.t. `E = E_base + scale·norm(P)` minimiert Diffusions-Noise-Loss zu den 2–5 Zielbildern:
   * **Optimizer:** AdamW auf {P, log_scale}
   * **LR:** 5e-3 (bei Instabilität 2e-3)
   * **Steps:** 600 (Range 400–800)
   * **Batch:** 2 (aus den 2–5 Bildern gezogen)
   * **CFG:** 4.5 (Range 3.5–6.0; senken bei Instabilität)
   * **Grad-Clip:** 1.0
   * **Target:** MSE auf ε oder v entsprechend `scheduler.prediction_type`
   * **Regularisierung:** Off-slot-Sparsity (oder harte On-the-fly-Maskierung) + leichte RMS-Reg
3. **Delta:** `Δ = E - E_base`. **Maskiere** `Δ` mit `M_slot` → speichere `R_teacher_slot = Δ[s:s+L,:]` (slot-reines Residual).
4. **QA:** Mini-Grid (Alpha-Sweep) mit α∈{0.5,1.0,1.5,2.0} ausgeben.

**Akzeptanz:**
* Loss fällt; scale=exp(log_scale) landet nicht an Clamp-Grenze.
* Off-slot-RMS ~ 0; Slot-RMS im vernünftigen Bereich.
* >80 % Subjekte bestehen visuelle QA (Identität erkennbar) bei α∈[1.0,1.5].

### T1 – Regressor (global)

* **Input:** (z, R_teacher_slot).
* **Backbone:** Qwen3-VL (frozen), **Feature-Extraktion:**
  * Vision-Tower, letzte Hidden-States
  * Mean-Pooling über alle Vision-Tokens
  * Mean über 1–3 Bilder
  * L2-Normalisierung: `z ← L2Norm(mean_images(mean_tokens(H_last)))`
  * **Pixel-Budget:** Gesamt ≤2.0 MP, max_side ≤1536 (proportionaler Downscale)
  * **Akzeptanz:** ‖z‖₂ ≈ 1.0 (±1e-3), dtype/device match global, keine VRAM-Spitzen
* **Head:** Linear-Baseline (D→L·H), **per-token RMS-Norm + exp(log_scale) Clamp [0.01,1.5]**.
* **Loss:** (0.7)·Cos + (0.3)·MSE + (0.05)·RMS-Alignment; optional **Render-Loss (low-res, selten)** für funktionale Erdung.
  * **Render-Loss-Frequenz:**
    * Jede Epoche: K=32 Identities, Low-Res (256px), 8–12 Steps, α=1.2 (nur Validation, nicht Training)
    * Optional: alle 1000 Updates 1 Mini-Batch (4–8 Identities) rendern
    * Metriken: Face-ID/Perceptual als Trend, **kein hartes Gate**
* **Eval:**
  * Offline: Cos-Sim zu Teacher-Slot ≥0.85, RMS-Fehler < 10 % (**hartes Gate**)
  * Online: Stichproben-Render (Alpha-Sweep) → visuelle Pass-Rate ≥80 %.

# 8) Inferenz (Runtime) – Schrittfolge

1. **Prompt mit Slots schreiben** (1–n Personen):
   *“boy riding a bike with ~ID42~ and ~ID7~ at sunset”*.
2. **Tokenisieren** → Slot-Fenster ([s_k:s_k+L_k)) je ~IDk~ finden.
3. **Base-Embeddings:** `E_base = encode(prompt)`.
4. **Residuals beschaffen:**

   * **Variante a)** (amortisiert): pro ~IDk~ Referenzbilder → Qwen3-VL-Features → Head → `R_k` ([L_k,H]).
   * **Variante b)** (direkt): vorhandenes `teacher_prompt_embeds.pt` + Maskierung → `R_k`.
5. **Injection:** `E_base[:, s_k:s_k+L_k, :] += α_k · R_k` (je Person eigene α, Standard 1.2).
6. **Qwen-Image aufrufen** mit `prompt_embeds=E_base` (+ optional `negative_prompt_embeds`).
7. **(Optional) Grids:** Seeds×Alphas zum schnellen QA.

# 9) Randfälle & Checks

* **Slot nicht gefunden:** harter Fail mit klarer Meldung („Slot ~ID~ nicht gefunden; Tokenizer/Version/Prompt inkonsistent"); Log + Abbruch.
* **Token-Länge weicht ab:** **Fixe Slot-Länge L (default: 4) ist zentral:**
  * L ist aus der Tokenisierung von ~ID~ unter dem gepinnten Tokenizer abgeleitet.
  * T0 & T1 & INF nutzen **exakt dieselbe** fixe Länge L.
  * Wenn ein Update am Tokenizer L ändert → **hartes Fail** mit Meldung, **kein** leises Pad/Trunk.
  * **(Notfall-Kompatibilität, nicht empfohlen):** Falls variabel unbedingt nötig:
    * L_ref beibehalten (z.B. 4)
    * Bei L_infer > L_ref: Repeat-Pad oder linear upsample entlang Token-Achse (artefaktträchtig)
    * Bei L_infer < L_ref: zentriert truncaten
    * **Besser:** Fixe Länge L nutzen und Tokenizer/Version einfrieren.
* **Mehrfach-Slots derselben ID:** gleiche `R_k` auf jedes Fenster.
* **Prompt-Konflikte (z. B. "boy" vs. erwachsene Person):** Warnen; α reduzieren, Prompt harmonisieren.
* **Mehrere Identities dicht beieinander:** min. 1 Token Abstand; sonst Greedy-Match der längeren Slot-Sequenz zuerst.

# 10) KPIs / DoD

* **Offline:** Top-1 Cos-Sim( (\hat R), (R^\text{teacher}_{slot}) ) ≥0.85; RMS-Abweichung <10 %.
* **Online:** ≥80 % visueller QA-Pass bei α∈[0.8,1.8]; Multi-ID-Prompts ohne gegenseitige Auslöschung.
* **Stabilität:** Kein sichtbarer Identitätsverlust bei Änderung von Stil/Ort/Action im Text.

# 11) Roll-out / Betrieb

* **Precompute** T0 für alle 20k (über Nacht / Cluster), speichere maskierte Slot-Residuals.
* **Train** T1 (1–2 Tage, Linear→MLP Feinschliff).
* **Serve**: Inferenz-API nimmt (Prompt mit Slots, 1–n Bilder pro ~ID~ oder fertige Teacher) → gibt Bild/Grids zurück.
* **Monitoring:** Anteil „PASS“ in Alpha-QA, Verteilung der optimalen α, Cos-Sim-Drift.
