# MushroomClassifier

## 1 - Prosjektbeskrivelse
### 1.1 - Omfang

Hensikten med dette prosjektet er å bygge et datasystem som hjelper sopplukkere å vurdere om en sopp er trygg å spise. Feilvurderinger kan få livstruende konsekvenser, og det kreves i dag et nivå av ekspertise i vurderingen av soppens spiselighet. Ved bruk av maskinlæring på detaljert morfologisk informasjon (hattsjerm, farge, sportrykk, ringtype, osv.) kan store deler av beslutningen automatiseres, og gi raske, konsistente svar. 

Systemet skal presenteres som et webgrensesnitt (Streamlit) der brukeren fyller inn kjente egenskaper, og får tilbake en sannsynlighetsvurdering for "spiselig" versus "giftig". Treningsdata hentes fra Kaggle ("Mushroom Edibility Classification") med over 60 000 syntetisk genererte, men biologisk plausible observasjoner. 

Prosjektet planlegges gjennomført ved hjelp av egen maskin og Streamlit Cloud for hosting. 

### 1.2 - Metrikker

Maskinlæringsmålet er høy presisjon og recall for klassen "giftig", slik at faren for å anbefale giftige sopper som spiselige er minimal. Total nøyaktighet følges opp, og ROC-AUC for helhetsbildet; modellen fra RandomForest-søket leverer 100% på alle metrikker på holdout-settet. 
"Business metric" er definert som null feilklassifisering av giftige prøver - èn eneste feil kan ha alvorlige konsekvenser. I praksis betyr dette at det kreves minst 99,9% recall på testsett og et UX som tydelig advarer om usikkerhet. 

En vellykket løsning kan fungere som en rådgivende tjeneste som sparer tid for eksperter, og kan ta rollen som et støttevalg for amatører. 
## 2 - Data
### 2.1 - Datasett

Datasettet *Mushroom Edibility Classification* fra Kaggle (*Zohaib Dev*, basert på *Dennis Wagners* genererte "*secondary_data.csv*") brukes. Filen er separert med semikolon og inneholder 61 069 syntetiske observasjoner med 20 egenskaper: 3 numeriske (hattdiameter i cm, stilklengde i cm, stilkbredde i mm) og 17 kategoriske (bl.a. hattform, overflate, farger, sporefarge, ringtype, habitat, sesong). Klassen er binær: ``e`` (spiselig) eller ``p`` (giftig/ukjent). 

Metadata (*secondary_data_meta.txt*) dokumenterer at datasettene er simulert ved å randomisere verdier fra 173 arter beskrevet i *Hardin (1999)*; det finnes derfor ingen persondata å bekymre seg for. 

Data hentes ved å laste ned Kaggle-arkivet og pakke ut *secondary_data.csv* til prosjektroten. Datasettet er komplett, men ``?`` og tomme strenger behandles som manglende verdier. Det er ikke forventet å måtte samle inn mer data, men prosjektet kan skaleres ved å kjøre samme preprosessering på andre soppdata. 
Siden målet er en klassifiseringsmodell med over 100 000 rader tilgjengelig (etter syntetisk generering) er datamengden tilstrekkelig; det deles 80/20 i trening/test og stratifiserer på klasse for å bevare forholdet ca. 50/50. 

### 2.2 - Modellrepresentasjon

Numeriske felt beholdes som flyt med median-imputering og standardisering. Nominale variabler imputeres ved modus og "one-hot" enkodes via ``OneHotEncoder(handle_unknown='ignore')``, slik at pipeline tåler nye kombinasjoner i bruk. Feature importance spores fra den trenede RandomForest-modellen for å forstå hvilke felt som driver beslutning. 

Datasettet er syntetisk, men produktet må kommunisere at modelldommer ikke erstatter feltbiolog eller giftinformasjon - en feilklassifisering kan være kritisk. Det advares tydelig i Streamlit-grensesnittet og produktet anbefales kun i bruk som et støtteverktøy. 

## 3 - Modellering

### 3.1 - Baseline og enkle modeller

Startet med en naiv baseline - alltid predikere majoritetsklassen (ca. 51% giftig) - for å ha et bunnivå. Testet en rask ``LogisticRegression`` med standard one-hot encoding for å bekrefte at dataene er separerbare (ga ~99% nøyaktighet). Resultatene ga trygghet for at mer komplekse modeller kunne gi 100% uten overtreningstendenser, ettersom dataene er syntetisk skapt for klar separasjon. 

### 3.2 - Hovedmodell

RandomForestClassifier inne i en ``sklearn``-pipeline: 

- ``ColumnTransformer``: 
  median-imputering og standardisering for numeriske
  modus-imputering og ``OneHotEncoder(handle_unknown='ignore')`` for kategoriske
- Randomforest med ``class_weight='balanced_subsample'`` for robusthet; søk i hyperparameterrom for ``n_estimators``, ``max_depth``, ``min_samples_split``, ``min_samples_leaf``, ``max_features``. 
- RandomizedSearchCV (3-fold, 50 kandidater) med ``f1_macro`` som scoring; ``n_jobs=1`` for sikker Windows-kjøring. 
- Beste konfigurasjon: ``n_estimators=600``, ``max_depth=none``, ``min_samples_split=2``, ``min_samples_leaf=1``, ``max_features=0.4``. 

### 3.3 - Evalueringsoppsett

Stratified train/test-splitt (80/20) med random seed 42. Resultat: 100% på accuracy, recall og precision for begge klasser samt ROC-AUC=1.0 - den syntetiske dataen gjør dette mulig. 
``confusion_matrix.png`` og ``metrics.json`` lagrer resultatene automatisk. 

### 3.4 - Feil- og featureanalyse

Ingen feil har så langt blitt observert. 

``feature_importances.csv`` viser mest innflytelsesrike dummy-variabler (stilkfarge, sporetrykk, gill-spacing, m.m.). Eventuelle fremtidige feil kan analyseres via ``classification_report.txt`` og ``cv_results.csv`` for å se hvilke hyperparametere som ga dårligere score. Det anbefales fortsatt manuell gjennomgang av feilklassifiserte eksempler når vi får nye data fra virkelige feltsituasjoner. 

### 3.5 - Forbedringsplan

Hvis modellen skal brukes til virkelige feltdata, planlegges testing med ekte observasjoner, eventuell oppdatering av pipeline til gradient boosting eller TabNet for bedre håndtering av blandede variabler, og aktiv feilovervåkning (lagring av input/score). I tillegg kan vi vurdere kalibrering (``CalibratedClassifierCV``) for sannsynlighetstolkning og en ensemblestrategi med enklere modeller for bedre forklarbarhet. 
## 4 - Deployment

### 4.1 - Web-applikasjon

Streamlit-app (``streamlit_app.py``) er frontenden. Den laster ``mushroom_classifier.joblib`` via ``@st.cache_resource``, viser skjema med sliders/dropdowns og gir prediksjon og sannsynligheter. En advarsel minner brukere på at modellen kun er rådgivende. 

### 4.2 - Tilgjengeliggjøring

Planlagt hosting på *Streamlit Community Cloud*. Repositoryet må inneholde ``requirements.txt``, ``streamlit_app.py``, ``models``-artefakter og datasett hvis vi ønsker offline-modus (60 000 rader ~9MB, under Cloud-grensen). Datasettet kan alternativt legges på Github Release og lastes ned ved app-start. 

### 4.3 - Arbeidsflyt

1. Tren lokalt med ``python src/train_model.py`` for å generere ``models``-filene (``mushroom_classifier.joblib``, ``best_params.json``, ``cv_results.csv``, ``metrics.json``, ``confusion_matrix.png``, ``feature_importances.csv``, ``training_summary.txt``). 
2. Push repository til Github og deploy. Streamlit Cloud har CI som installerer avhengigheter og starter appen. 

### 4.4 - Monitorering og vedlikehold

I produksjon blir brukerforespørsler loggført (uten persondata) for å samle mislabeled tilfeller, trigger regelmessig ny trening hvis nye data kommer, og overvåker grep som recall på "poisonous". ``cv_results.csv`` gjør det lett å gjenopprette beste parametre. 
## 5 - Referanser

- Dev, Z. (2024). _Mushroom Edibility Classification_ [Datasett]. Kaggle. [https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification]
- Wagner, D. (2020). _Secondary mushroom data_ (metadata og Python-skript). [https://mushroom.mathematik.uni-marburg.de/files/]
- Hardin, P. (1999). _Mushrooms & Toadstools_. Zondervan Publishing. 
- Schlimmer, J. (1987). _Mushroom Data Set_. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Mushroom]
- Pedregosa, F., et al. (2011). _Scikit-learn: Machine Learning in Python_. Journal of Machine Learning Research, 12, 2825‑2830. 
- Streamlit Inc. (2024). _Streamlit Documentation. [https://docs.streamlit.io/]
