# M1-KRBF: experiment exploratori amb configuració fixa

KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0), quatre sortides, sense cerca.
La mitjana dels targets es calcula només sobre els congruents que ajusten cada
model; s'ajusta sobre P_A - mitjana i es restitueix la mitjana després de predir.
No es centren les entrades. Es comproven quatre components finites i suma 1
amb tolerància absoluta 1e-6. No s'aplica clipping ni renormalització. Es desen
prediccions crues, components fora [0,1] i error de suma. El centrat permet
conservar residuals de suma aproximadament zero i les tres primeres coordenades.

Es reutilitzen exclusivament m1_audio_train_oof.csv, m1_residual_targets.csv i
m1_partitions.json. Es validen cobertura per scope i model_fit_id, grups,
etiquetes, rol, probabilitats, exclusions de test extern i de validació estricta
en tots els SVM/calibradors registrats. No s'ajusta cap SVM ni es recalculen P_A.

Per fold extern: cinc models condicionals de referència, ajustats només amb
congruents dels seus train, i un model final amb tots els congruents del train
extern. Targets de referència acústics exclosos dels textos reservats. El P_A
observat de validació ve de la taula OOF base. Es recalculen residuals i
Ledoit–Wolf, sense reutilitzar residuals/covariàncies de Ridge lineal.
Total: **30 ajustos Kernel Ridge, 5 covariàncies, 0 ajustos SVM**.

El test es copia de m1_scores.csv, contrastat amb B1, manifest i splits; igualtat
de vectors en memòria. Després de serialitzar CSV poden aparèixer arrodoniments
de lectura: concordança relativa 1e-14, sense tolerància absoluta, amb B1.
El score és Mahalanobis al quadrat centrat, sobre angry/happy/neutral, mitjançant
les funcions originals M1. Es requereix covariància definida positiva.

AUROC/AP per fold i pooled de B1, M1 i M1-KRBF; sense llindar binari. Dos
bootstraps aparellats (KRBF-M1, KRBF-B1), cadascun 10.000 rèpliques, seed=42,
grups textuals sencers amb reemplaçament, conservant multiplicitat. Percentils
2.5/97.5; rèpliques invàlides registrades sense substituir. Els models queden fixos.

Sortides exclusives m1_krbf_: run_config.json, scores.csv,
reference_residuals.csv, models.json (mitjanes, dual_coef, X_fit, covariàncies,
identificadors d'ajust/validació i diagnòstics), metrics.csv,
bootstrap_summary.csv, bootstrap_replicas.csv i audit.json. Tots els artefactes
previs queden protegits i els seus hashes es comproven abans d'exportar.
El pla reutilitzat queda identificat pel hash de m1_partitions.json.

Execució des de l'arrel:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_m1_kernel_rbf.py -v
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/m1_kernel_rbf.py
```

Limitacions: experiment proposat després de veure els resultats S1; configuració
fixa exploratòria no optimitzada, sense validació independent. OOF independent
d'ajust/calibratge, però no de selecció prèvia d'hiperparàmetres SVM. Diferències
de mida entre ajustos de referència/finals. P_T normalitzat no implica calibratge.
Mahalanobis no és probabilitat de mismatch. L'objectiu continua sent congruència
sintètica EMIS, no percepció humana. No es pressuposa superioritat de KRBF.

## Execució verificada (2026-11-10)

Completats 30 ajustos Kernel Ridge, 5 covariàncies i cap ajust SVM nou.
AUROC pooled KRBF=0.944975, AP=0.975245. Diferències respecte de M1:
AUROC +0.004441, IC95% [0.000284, 0.011759]; AP +0.002241,
IC95% [0.000207, 0.005627]. Respecte de B1: AUROC -0.010221,
IC95% [-0.022980, 0.003990]; AP -0.004726, IC95% [-0.014419, 0.007084].
Cada comparació té 10.000 rèpliques vàlides i cap invàlida.

Cap sortida de test ni de referència fora [0,1]. Error màxim de suma en test:
2.67e-15. Les cinc covariàncies són definides positives. Verificats els
hashes dels artefactes previs, les particions estrictes, les mitjanes de
centrat, les prediccions reconstruïdes amb kernel i coeficients duals, els
scores Mahalanobis i els percentils dels intervals exportats.
Passen 38 tests descoberts i les 3 comprovacions de splits.

La millora petita respecte de M1 és exploratòria posterior als resultats S1;
no es presenta com una confirmació independent de superioritat. Els valors
puntuals globals continuen afavorint B1.
