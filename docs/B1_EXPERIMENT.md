# B1 — S1, WavLM/SVM-RBF i qwen3_4b

## Estat

Implementació preparada per a tests i auditoria. No s'ha entrenat sobre EMIS.
L'usuari ha validat sigmoid agrupat + SVM final amb tot el train extern,
ensemble=False, resolent la decisió pendent del protocol §10.1 per S1.
El CLI exigeix `--calibration-ensemble false` quan s'activa `--train`.
L'execució costosa continua pendent d'autorització.

## Fonts i selecció

Es comproven els hashes de les fonts contra `run_config.json` de B0, les
particions i la reproducció de les mètriques existents. Es reutilitza la
selecció de cada registre de `fold_models.json`, sense repetir la cerca ni
consultar les mètriques de test per seleccionar res. Actualment: pooling
mean, capa 3, C=10, gamma=0.001 en els cinc folds. Els models B0 no estan
serialitzats: cal reajustar SVM i scaler des dels embeddings existents.
La discrepància històrica del hash del manifest i el commit WavLM desconegut
es mantenen documentats a `B0_PRELIMINARY.md` i a l'auditoria d'entrada.

## Calibratge validat i alternativa considerada

Sigmoid (one-vs-rest amb normalització multiclasse de scikit-learn), amb
cinc particions `StratifiedGroupKFold`, shuffle=True, seed=42, grup=group_text,
estratificació per generation_emotion; sempre només dins del train extern.
Cada ajust de l'SVM inclou un StandardScaler ajustat només amb el seu train.
No s'activa `SVC(probability=True)` ni una CV interna sense grups.

- Opció validada, `ensemble=false`: cinc SVM generen scores OOF dels
  exemples de train. El calibrador aprèn d'aquests scores i les etiquetes
  de generation_emotion de train. Un sisè SVM s'ajusta amb tot el train extern
  i el calibrador transforma els seus scores del test. 30 ajustos SVM totals.
- Alternativa considerada, no seleccionada, `ensemble=true`: cinc parelles SVM/calibrador, cadascuna amb
  grups disjunts d'ajust/calibratge; al test es fa la mitjana de les seves
  probabilitats. 25 ajustos SVM totals. Canvia el predictor final a un ensemble.

En tots dos casos la selecció d'hiperparàmetres heretada de B0 ha vist tots
els grups del train extern, inclosos els grups de calibratge. Els scores són
OOF respecte de l'ajust de cada SVM, però no respecte de la selecció de
configuració. No es presenta aquest esquema com una estimació independent
de qualitat de calibratge al train. El test extern queda completament fora.
Amb ensemble=false també pot haver-hi canvi de distribució dels scores entre
els SVM parcials i el SVM reajustat amb tot el train. Són limitacions a registrar.

## Scores i avaluació

JS és la divergència, no la distància arrel quadrada: log base 2, zeros
tractats amb 0 log 0 = 0, sense epsilon, clipping ni renormalització de P_T.
Quatre components finits dins [0,1], suma amb tolerància absoluta 1e-6.
Classes acústiques reordenades explícitament des de classes_. Qwen continua
sent una distribució normalitzada, sense garantia de calibració estadística.

AUROC i AP per fold i pooled. Balanced accuracy i F1 B1 queden absents amb
estat pending_threshold; no es tria cap llindar sobre el test. Comparació
amb B0 només sobre claus i ground truth exactament idèntics. Com que B0
és preliminar i sense calibratge, la diferència no aïlla exclusivament l'efecte
de JS. Es registra el canvi d'etiqueta acústica dominant respecte de B0.
No es modifica B0 ni es calcula un nou baseline categòric. No s'implementa M1.

## Sortides previstes

Fitxers nous sota outputs/mismatch_v1, amb creació exclusiva:
b1_run_config.json, b1_scores.csv (P_A, P_T, y_mismatch, b1_score, claus,
identificador d'ajust, B0 aparellat), b1_metrics.csv, b1_comparison.csv,
b1_fold_models.json (identificadors de train/calibratge/test), b1_audit_report.json.
Es conserven versions de llibreries, llavors, revisions Qwen, hashes de codi,
fonts i tots els artefactes B0; es verifica que B0 no ha canviat abans d'exportar.

Auditoria sense entrenament:

```powershell
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/b1_js.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_b1_js.py -v
```

Comanda preparada, pendent d'autoritzar entrenament (modalitat completa no executada):

```powershell
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/b1_js.py --train --calibration-ensemble false --n-jobs 1
```
