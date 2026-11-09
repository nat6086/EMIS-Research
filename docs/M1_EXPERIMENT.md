# M1 S1 — especificació implementada, pendent d'execució

Decisions validades per l'usuari: WavLM mean, capa 3, SVM-RBF C=10,
gamma=0.001, calibratge sigmoid agrupat ensemble=False; Ridge alpha=1.0
amb intercept; residuals estrictament fora de mostra; Ledoit–Wolf sobre
les tres primeres components; Mahalanobis al quadrat centrat. No hi ha
cerca d'hiperparàmetres, clipping, renormalització ni llindar de mismatch.

## Jerarquia exacta i recompte

Per cada fold extern S1 es reserva el test existent T i s'anomena A el train.
Tots els splits interns tenen cinc particions, seed=42, shuffle=True,
StratifiedGroupKFold estratificat per generation_emotion i agrupat per group_text.
Les particions es construeixen amb tots els àudios del subconjunt; les màscares
de congruència només s'apliquen quan s'ajusta Ridge o la covariància.

1. A es divideix en cinc particions (A\V_j, V_j). Per cada j s'ajusta una
   cadena acústica calibrada només en A\V_j i es prediu V_j. La concatenació
   forma els targets acústics OOF per al Ridge final.
2. Per obtenir residuals estrictes de Ridge a V_j, es divideix de nou A\V_j
   en cinc particions OOF. Cadascuna rep probabilitats d'una cadena acústica
   ajustada i calibrada sense aquella partició i sense cap text de V_j.
   Aquests targets serveixen per ajustar Ridge només amb congruents d'A\V_j.
3. Ridge prediu P_A esperat dels congruents de V_j. El P_A observat per aquests
   exemples es reutilitza del pas 1: tampoc ha vist V_j durant ajust/calibratge.
   Els residuals dels cinc V_j són la referència congruent per Ledoit–Wolf.
4. El Ridge final s'ajusta amb tots els congruents d'A i els targets del pas 1.
   El test reutilitza directament P_A i P_T de b1_scores.csv. No es reajusta
   l'acústica per al test.

Cada cadena acústica fa cinc ajustos SVM per scores OOF de calibratge i un
reajust final amb tot el seu train. El scaler forma part de cadascun d'aquests
ajustos. Els calibradors sigmoid només veuen scores i etiquetes del train
d'aquella cadena; mai V_j quan està reservat, ni T.

| Component | Per fold extern | Cinc folds |
|---|---:|---:|
| Targets OOF del Ridge final: 5 cadenes × 6 SVM | 30 | 150 |
| Targets per residuals estrictes: 5 × 5 cadenes × 6 SVM | 150 | 750 |
| Total SVM | 180 | **900** |
| Ridge (5 de validació i 1 final) | 6 | 30 |

Es reutilitzen les probabilitats de validació del pas 1. Sense aquest reús
caldria recalcular-les innecessàriament. El recompte es calcula des del pla
real de particions i té un test automatitzat. No es compta l'auditoria com
a entrenament: el mode sense --train només carrega dades i construeix splits.

## Residuals i limitacions

Es conserven quatre sortides Ridge crues. La suma es comprova amb atol=1e-6;
els valors fora de [0,1] s'exporten i es compten, sense modificar-los. Els
residuals són P_A - P_A_esperat. Ledoit–Wolf estima mitjana i covariància només
amb els residuals congruents cross-fitted, coordenades angry/happy/neutral.
Es comproven autovalors positius; una covariància degenerada atura l'execució.
El score és (r3-mu)' Sigma^-1 (r3-mu), resolt sense invertir directament Sigma.

Els targets són OOF respecte de l'ajust i calibratge, però les configuracions
heretades B0/B1 van ser seleccionades amb tot A: no són OOF respecte d'aquesta
selecció prèvia. Aquesta limitació s'ha acceptat explícitament.
Els SVM i Ridge parcials s'ajusten amb menys dades que els finals; això pot
canviar la confiança acústica i la distribució dels residuals. S'exporten
mitjanes acústiques OOF/test i residuals de referència per auditar-ho; no es
fan correccions seleccionades després de veure el test.

La lectura dels CSV Qwen i B1 pot diferir en arrodoniments de serialització
(s'ha observat fins a 2.65e-23). Es valida la concordança relativa a 1e-14,
sense tolerància absoluta, i s'utilitzen les components textuals desades a B1
per mantenir la mateixa representació. Els vectors de test es copien de B1
i se'n comprova la igualtat exacta dins de l'execució.

Mahalanobis no és una probabilitat calibrada. El ground truth continua sent
la condició de generació sintètica, no un judici perceptual humà. No es
pressuposa superioritat de M1.

## Artefactes i comprovacions

Entrada: src/emis_research/mismatch/m1_conditional.py. Totes les sortides són
noves sota outputs/mismatch_v1, prefix m1_, mode de creació exclusiu:

- m1_run_config.json: versions, hashes, decisions, recompte i llavors.
- m1_partitions.json: identificadors de train/test extern, splits Ridge,
  cada train/predict acústic, i cada train/validació de calibratge.
- m1_audio_train_oof.csv: targets del Ridge final, rol train_oof per fold.
- m1_residual_targets.csv: targets dels Ridge de referència, identificats
  per scope i model_fit_id; un àudio pot repetir-se en diferents scopes.
- m1_reference_residuals.csv: només congruents, prediccions crues i residuals.
- m1_scores.csv: test B1 intacte, sortides Ridge, residuals i score M1.
- m1_models.json: coeficients Ridge, mitjana/covariància, shrinkage,
  identificadors congruents i diagnòstics; no serialitza SVM executables.
- m1_metrics.csv: AUROC/AP B1/M1 per fold i globals.
- m1_bootstrap_summary.csv i m1_bootstrap_replicas.csv: 10.000 rèpliques,
  seed=42, grups de text amb reemplaçament, percentils 2.5/97.5, diferència
  M1-B1. Rèpliques invàlides registrades sense substituir. Scores fixos,
  sense incorporar variabilitat de reentrenament als intervals.
- m1_audit_report.json: cobertura i conservació dels artefactes anteriors.

Comandes d'auditoria i tests, sense entrenament sobre EMIS:

```powershell
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/m1_conditional.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_m1.py -v
```

Comanda preparada però NO executada; requereix autorització dels 900 ajustos:

```powershell
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/m1_conditional.py --train --n-jobs 1
```
