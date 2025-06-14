# Project Implementation Plan

## Ziel
Dieses Dokument enthält einen detaillierten Plan zur Umsetzung des "Modern Edge Detection Toolkit" gemäß README.md.

## Hauptaufgaben
1. **Projektstruktur anlegen**
   - Verzeichnis `modern_edge_toolkit/` mit den Unterordnern und Dateien wie in der README beschrieben.
   - Unterordner für Submodule (`bdcn_repo`, `rcf_repo`, `hed_repo`, `dexined_repo`, `casenet_repo`).
   - Ordner `models/structured/`, `images/`, `results/` und eine zentrale `config_modern.yaml`.

2. **Submodule integrieren**
   - `git submodule add https://github.com/zijundeng/BDCN.git bdcn_repo`
   - `git submodule add https://github.com/yun-liu/RCF.git rcf_repo`
   - `git submodule add https://github.com/s9xie/hed.git hed_repo`
   - `git submodule add https://github.com/csyanbin/DexiNed.git dexined_repo`
   - `git submodule add https://github.com/cvlab-yonsei/edge_detection.git casenet_repo`
   - Modelle aus den jeweiligen Repositories herunterladen und in die angegebenen Ordner speichern.

3. **Gewichtedateien herunterladen**
   - BDCN: `bdcn_repo/pretrained/bdcn_pretrained.pth`
   - RCF: `rcf_repo/model/RCF.pth`
   - HED: `hed_repo/hed_pretrained_bsds.caffemodel` und `hed_repo/deploy.prototxt`
   - DexiNed: `dexined_repo/weights/dexined.pth`
   - CASENet: `casenet_repo/model/casenet.pth`
   - Structured Forest: `models/structured/model.yml.gz`

4. **Python-Abhängigkeiten verwalten**
   - `requirements.txt` anlegen mit den in der README genannten Paketen (torch>=2.0.0, torchvision, opencv-python usw.).
   - Installation via `pip install -r requirements.txt`.

5. **`detectors_modern.py` erstellen**
   - Lade- und Inferenzfunktionen für jede der sechs Methoden.
   - GPU-Nutzung via PyTorch sicherstellen.
   - Gemeinsame Schnittstelle definieren, um Bilder einzulesen, Ergebnisse zu speichern und Speicher zu bereinigen.

6. **`gui_modern.py` implementieren**
   - Streamlit-Oberfläche mit folgenden Elementen:
     - Ordnerauswahl für Eingabebilder.
     - Checkboxen für jede Methode (BDCN, RCF, HED, DexiNed, CASENet, Structured Forest).
     - Button "🚀 Verarbeitung starten".
     - Fortschrittsanzeige (z.B. über `st.progress` und `tqdm`).
   - Ergebnisse werden in `results/{METHODENNAME}/` abgelegt.

7. **Batch-Verarbeitung**
   - Alle Bilder des gewählten Eingabeordners in unterstützten Formaten (PNG, JPG, TIF, …) verarbeiten.
   - Keine Vorschau oder interaktive Anzeige.

8. **Konfiguration**
   - Parameter in `config_modern.yaml` definieren (z.B. Pfade, GPU-Einstellungen, batch size).

9. **Speicher- und Ressourcenmanagement**
   - Vor und nach jeder Methode GPU-Speicher freigeben (`torch.cuda.empty_cache()`).
   - Nutzung von `psutil` zur Überwachung optional vorsehen.

10. **Testing und Qualitätssicherung**
    - Einfache Unit-Tests für Hilfsfunktionen in `detectors_modern.py`.
    - Funktionsprüfung der Streamlit-GUI durch manuelles Starten (`streamlit run gui_modern.py`).

11. **Dokumentation**
    - README um Schritte zur Installation und Nutzung ergänzen.
    - Hinweise zu benötigter Hardware (GPU-Unterstützung) und zu den Speicheranforderungen der Modelle.

## Hinweise
- Es sind **keine** klassischen Edge-Detectoren wie Canny oder Sobel zu integrieren.
- Kein Docker, keine CLI-Skripte und keine Vorschau-Funktion.
- Das System arbeitet ausschließlich über die GUI.

