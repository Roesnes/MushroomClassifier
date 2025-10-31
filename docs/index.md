# MushroomClassifier

Velkommen til MushroomClassifier-prosjektet. Denne siden beskriver hvordan du utforsker dataene, reproduserer maskinlæringspipelinen og bruker webapplikasjonen.

---

## Prosjektoversikt

- **Mål:** Forutsi om en sopp er spiselig eller giftig basert på morfologiske kjennetegn.
- **Teknologistakk:** pandas, scikit-learn, Streamlit, matplotlib.
- **Datasett:** [Mushroom Edibility Classification (Kaggle)](https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification)
- **Modell:** RandomForest-pipeline med one-hot-koding og automatisert hyperparametersøk. Nåværende beste modell oppnår 100 % presisjon/recall på holdout-settet (syntetiske data).

> ⚠️ Dette verktøyet er kun ment som beslutningsstøtte. Søk alltid råd hos autoritative kilder før du spiser sopp fra naturen.

---

## Kom i gang

1. **Klon repositoriet**
   ```powershell
   git clone https://github.com/<din-bruker>/MushroomClassifier.git
   cd MushroomClassifier
   ```
2. **Opprett virtuelt miljø**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Installer avhengigheter**
   ```powershell
   pip install -r requirements.txt
   ```
4. **(Valgfritt) Tren modellen på nytt**
   ```powershell
   python src/train_model.py --data-path secondary_data.csv --output-dir models
   ```
5. **Start Streamlit-appen**
   ```powershell
   streamlit run app/streamlit_app.py
   ```

---

## Nøkkelartefakter

- `models/mushroom_classifier.joblib` — Serialisert pipeline klar for inferens.
- `models/metrics.json` — Presisjon/recall/F1 og ROC-AUC for testsettet.
- `models/feature_importances.csv` — Rangerte feature-importanser for tolkbarhet.
- `models/cv_results.csv` — Resultater fra hyperparametersøket for reproduserbarhet.
- `app/streamlit_app.py` — Webgrensesnitt for interaktive prediksjoner.
- `src/predict.py` — CLI-verktøy for inferens (JSON inn/ut).

---

## Utrullingsnotater

- En-klikks utrulling kan gjøres med [Streamlit Community Cloud](https://share.streamlit.io). Sett startfil til `app/streamlit_app.py` og sørg for at `requirements.txt` finnes.
- Inkluder `models/`-artefaktene i repositoriet slik at appen laster raskt. Eventuelt kan de genereres ved første oppstart (tyngre beregning).
- Foretrekker du statisk hosting, kan du bygge inn Streamlit-appen med et iframe på denne siden etter utrulling.

---

## Dokumentasjon og rapportering

- **EDA-sjekkliste:** `notebooks/eda_checklist.md`
- **Treningsoppsummering:** `models/training_summary.txt`
- **Prosjektrapport:** Bruk DAT158-malen og oppgi kildene som er listet i repositoriet.

For spørsmål eller bidrag, opprett et issue eller send inn en pull request på GitHub.
