# B0 preliminar — S1 / SVM-RBF / qwen3_4b

## Auditoria prèvia (2026-11-06)

El manifest actual té SHA-256
`5c38f86f6d555787ea63c2c4bcf78b384c516f62138e9cc4802944e312867df7`.
Coincideix amb `outputs/text_baseline/run_meta.json`, però les metadades
dels splits i WavLM registren
`8ca783c7107c4500d3d00b6abbb63efaeb031a712447cf2ca7db7bdca3cf0a51`.
No es disposa de la versió antiga per establir la causa exacta. Ni reordenar
les columnes segons `build_manifest.COLUMNS` ni canviar LF/CRLF reprodueix
el hash antic. No s'ha de presentar aquesta discrepància com a resolta.

Comprovacions executades abans de programar: reconstrucció en memòria des
del ZIP i CSV originals amb igualtat de totes les cel·les per `audio_id`;
1.248 identificadors únics idèntics en manifest, NPZ i splits; correspondència
exacta de `group_text`; reconstrucció exacta dels splits existents;
embeddings `mean` finits de forma (1248, 13, 768). No s'ha detectat una
incompatibilitat de contingut. El hash del NPZ coincideix amb les seves
metadades. `model_commit` WavLM és null: la revisió original continua desconeguda.

## Implementació i execució

Entrada nova: `src/emis_research/mismatch/b0_labels.py`. No altera scripts
originals. Mode per defecte: auditoria sense entrenament. `--train` executa
els cinc folds S1 existents, només SVM-RBF, sense probabilitats ni calibratge.
Reutilitza `make_model`, `fit_score`, `GRIDS`, `SEED` i `INNER_K` originals.
Pooling `mean`, capes candidates [3, 6, 9, 12], C [0.1, 1, 10, 100], gamma
[0.0003, 0.001, 0.003, 0.01]. Són els defaults del codi original; no s'afirma
que el pooling de l'execució històrica estigui acreditat.

64 configuracions × 5 folds interns × 5 externs + 5 ajustos finals =
1.605 ajustos SVM. Selecció per UAR interna, primer candidat en cas d'empat.
Escalat ajustat dins de cada train. No s'utilitza el test per seleccionar.

Qwen es reutilitza sense inferència nova, amb revisió i prompt registrats.
Argmax textual en ordre angry, happy, neutral, sad; es compten els empats.
L'etiqueta acústica és `SVC.predict`, amb la política de desempats pròpia
d'SVC (`break_ties=False`), no un argmax de probabilitats. Aquest B0 és
preliminar i caldrà homologar-lo amb els vectors futurs.

AUROC i average precision utilitzen el score binari. Balanced accuracy i
F1 utilitzen directament la decisió B0 (positiu = incongruent), sense ajustar
llindars al test. Folds sense dues classes: mètriques absents i causa explícita.
No s'estimen intervals que tractin els clips com a independents.

Sortides sota `outputs/mismatch_v1/`: `audit_report.json` abans d'entrenar;
després `audio_predictions.csv`, `text_predictions.csv`, `mismatch_inputs.csv`,
`mismatch_scores.csv`, `metrics_by_fold.csv`, `fold_models.json` i `run_config.json`.
Els fitxers es creen en mode exclusiu; una execució repetida falla abans
d'entrenar si alguna sortida prevista ja existeix. L'auditoria es desa una
sola vegada; `--train` exigeix que continuï coincidint amb les fonts actuals.
Les mètriques reals estan pendents d'autorització de l'entrenament.

Comandes verificades des de l'arrel (sense entrenar el corpus):

```powershell
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/b0_labels.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_b0.py -v
.\.venv\Scripts\python.exe -B tests/test_splits.py
```

L'entrenament només s'activa afegint `--train` a la primera comanda.
No s'ha executat aquesta modalitat sobre EMIS. `--n-jobs` és 1 per defecte
per limitar memòria i ús de CPU; no s'ha estimat el temps real.
