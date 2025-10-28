Inspiriert vom Textual Inversion Prinzip von Stable Diffusion (SDXL), möchte ich folgendes umsetzen:
Wir feintunen ein Qwen3-VL (https://github.com/QwenLM/Qwen3-VL) so, dass es aus 1-n Bildern einer Person, die Tokens vorhersagen kann, die Qwen Image (https://github.com/QwenLM/Qwen-Image), um neue Bilder mit dieser Person in unterschiedlichen Szenen, in unterschiedlichen Posen und mit unterschiedlichen Gesichtsausdrücken zu erzeugen. Wichtig ist, dass dabei der aktuelle Zustand der Person (Frisur, Schmuck, Bart, Brille) berücksichtigt werden und nicht reine Identität (im Gegensatz zu InsightFace und ArcFace).


Laufzeitumgebung ist modal.com . Die einzelnen Anwendungen werden über harald/modal_runner.py gestartet: z.B. `modal run -m harald.modal_runner::injection_poc`


---

## 1) Projekt-Annahmen & Goldene Regeln

- **Repo & venv existieren.** `harald/config.py` ist die _Single Source of Truth_ für Modal-/Runtime-Objekte, Versionen und Volumes.
- **Determinismus:** Gleiches Seed ⇒ gleiche Outputs (Train/Infer).
- **CIDA/GPU:** Utilize device CIDA and the GPU whereever possible. The code does not have to run on machines without CIDA.
- **modal.com:** Alle aufrufbaren Funktionen werden als modal.com remote functions implementiert. Aufrufschema siehe unten. Dateien, die gelesen werden müssen, werden per volume zur Verfügung gestellt. Modelle (Comic-Style-LoRA) auf dem HF_CACHE. Führe alle Tests auf modal.com aud, wie unten in "Jobs & Tests ausführen" gezeigt.
- **huggingface:** Beim Laden von Modelle, Pipelines, etc. über Huggingface nutze verpflichtend immer `token = hf_token` und `cache_dir=HF_CACHE`. Lies `hf_token` aus dem `HF_SECRET` falls noch nicht geschehen: `hf_token = os.environ.get("HUGGINGFACE_TOKEN")`

--

## 2) Trainingsdaten

* **Speicherort:** Die Trainingsdaten liegen verteilt in mehreren Basis-Verzeichnissen (`/mnt/dataset/1`, `/mnt/dataset/2`, etc.).
* **Identitäts-/Zustands-Gruppierung:** Jedes Basis-Verzeichnis enthält ca. 2000 Unterverzeichnisse.
Jedes Unterverzeichnis enthält Bilder und dazugehörige Labels zu einer Identität/Zustand. Der Name jedes Unterverzeichnisses folgt dem Muster `seed_<number>` (z.B. `seed_19456895`) und repräsentiert die **eindeutige `state_id`** für eine Identität/Zustand im spezifischen Zustand.
* **Dateien pro Identität/Zustand:** In jedem `seed_...`-Verzeichnis befinden sich die zugehörigen Bild- und Textdateien.
    * **Bilder:** Benannt nach dem Muster `domain_view_XX.png` (z.B. `comic_view_01.png`, `photo_view_08.png`).
    * **Prompts:** Zu jedem Bild existiert eine Textdatei mit demselben Namen und der Endung `.txt` (z.B. `comic_view_01.txt`).


## 3) Outputs

Sämtliche erzeugten Daten werden nach `/mnt/output` und dort in ein passendes Unterverzeichnis geschrieben.
