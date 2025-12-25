# Echtzeit-Verkehrszeichenerkennung mit YOLOv12

Dieses Repository enthält den wichtigsten Code für mein KI-Projekt zur Erkennung von Verkehrszeichen.

- **Modell:** YOLOv12n

## 📁 Repository-Struktur
- `scripts/`: Die relevantesten Skripte von der Datenvorbereitung bis zum Training.
  - `split.py`: Stratifizierter Datensatz-Split.
  - `refinement_split.py`: Ausgleich unterrepräsentierter Klassen.
  - `augmented_weather.py`: Simulation von Regen, Nacht & Blendung.
  - `final_training.py`: Das finale Trainings-Skript.
  - `realtime_inference.py`: Live-Demo & FPS-Messung.
- `models/`: Enthält die trainierte `best.pt`.
- `data.yaml`: Klassen-Definitionen.

**Demo-Video**
https://iubhfs-my.sharepoint.com/:v:/g/personal/finn_thomsen_iu-study_org/IQCfOFHYVnVjQK2NlH7HJofsARKLaYU1PofbY4w6_dIg-Ek?e=HjUaEB
