# AGENTS.md — Instruccions per a Codex (SkillzU / EMIS)

## Missió

Ajudar a implementar, validar i documentar una comparació reproduïble de tres mètodes per detectar **incongruència emocional acústica–semàntica** en el corpus sintètic EMIS:

- **B0 — desacord d'etiquetes:** `1[argmax(P_A) != argmax(P_T)]`.
- **B1 — divergència Jensen–Shannon:** `JS(P_A, P_T)` amb logaritme en base 2.
- **M1 — model de congruència condicional:** predir `P_A` esperat des de `P_T` amb Ridge entrenat només amb exemples congruents i puntuar-ne els residuals mitjançant una covariància regularitzada i Mahalanobis.

**Objectiu científic:** provar si modelar l'emoció acústica esperada condicionada pel contingut emocional textual aporta capacitat discriminativa i generalització respecte dels baselines directes. **No pressuposar mai que M1 millora B1.**

## Com ha de treballar l'agent

1. **Comunicar-se en català**, amb explicacions comprensibles per a una persona amb nocions bàsiques de ML. Mantenir noms d'API, fitxers i conceptes tècnics en anglès quan pertoqui.
2. **Primer inspeccionar l'estructura real del repositori.** El ZIP de referència conté els scripts dins de `emis_research/`; els scripts mateixos indiquen una ubicació prevista `src/emis_research/`. No canviar rutes ni traslladar fitxers a cegues; confirmar la disposició real i les rutes resoltes abans d'executar res.
3. **Una etapa cada vegada:** primer auditar/normalitzar entrades, després B0, després probabilitats i B1, finalment M1 i avaluació. Explicar què es modificarà i informar de les proves executades.
4. **No canviar unilateralment el protocol científic**, les etiquetes, els splits, els grups de CV, les definicions de scores o la selecció de models. Si cal una decisió metodològica, exposar opcions i demanar validació.
5. Fer canvis petits, revisables, amb **tests automatitzats** i sortides reproduïbles. Reutilitzar els scripts existents quan sigui correcte, però sense comprometre la traçabilitat dels resultats originals.
6. No afirmar resultats empírics que no s'hagin executat i verificat. No inventar pesos, àudios, mètriques, fitxers ni execucions.
7. El document normatiu per a les decisions és [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md); el context i l'estat del repositori són a [`README.md`](README.md).

## Context del codi existent (verificat al ZIP de referència)

Scripts existents: `build_manifest.py`, `make_splits.py`, `audio_wavlm_extract.py`, `audio_wavlm_predict.py`, `text_emotion_extract_predict.py`, `text_emotion_evaluate.py`, `audio_egemaps_extract.py`, `audio_egemaps_predict.py`, `audio_signal_extract.py`, `audio_signal_predict.py`.

- `build_manifest.py` prepara `data/processed/emis_manifest.csv` a partir d'àudios i textos que **no s'inclouen** al ZIP de codi.
- `make_splits.py` defineix S1 (5 folds per text), S2 (*leave-one-reference-out*) i S3 (*leave-one-TTS-out*) i preveu `configs/splits/splits_v1.csv`.
- `audio_wavlm_predict.py` implementa CV aniuada i SVM RBF entre diverses famílies; **actualment fa servir `.predict()`**, no exporta les probabilitats calibrades ni una taula d'OOF d'entrenament pròpia del M1.
- `text_emotion_extract_predict.py` genera puntuacions Qwen per emoció normalitzades entre 4 opcions; aquest fet **no garanteix calibració estadística**. Conservar també `option_mass` si està disponible.
- El ZIP inspeccionat **no conté** àudios, manifest generat, prediccions per àudio, resultats de B0/B1/M1 ni mètriques noves.

## Invariants científics — obligatori

### Etiquetes i procedència

- Ordre canònic: **`angry`, `happy`, `neutral`, `sad`**. Fixar-lo explícitament a codi i CSV; comprovar sempre l'ordre de les classes retornat per l'SVM (`classes_`).
- `P_A`: probabilitats acústiques de **WavLM + SVM-RBF + calibratge**, **4 valors**, no embeddings WavLM de dimensions elevades.
- `P_T`: quatre puntuacions emocionals normalitzades de Qwen per text; tractar la seva calibració com una limitació/decisió metodològica, no com un fet garantit.
- `y_mismatch = int(text_emotion != generation_emotion)` és **la condició de generació sintètica**, no una etiqueta de percepció humana. No fer afirmacions perceptuals sense una validació externa.
- B0, B1 i M1 han de rebre la **mateixa parella d'entrades acústica/textual** per a cada àudio i protocol; guardar exactament quina versió de cada model genera les prediccions.

### Validació i absència de fuga d'informació

- Respectar **els folds externs existents** per a S1/S2/S3. Mai entrenar, calibrar, seleccionar característiques, ajustar hiperparàmetres, regularitzar covariància o seleccionar un llindar utilitzant el test extern.
- Les prediccions acústiques emprades per entrenar Ridge han de ser **OOF respecte dels exemples d'entrenament**, amb particions i ajustos compatibles amb el protocol extern. No utilitzar prediccions *in-sample* com a objectius sense una justificació revisada.
- Entrenar Ridge i estimar la covariància **només amb exemples congruents de l'entrenament extern**. No fer servir els incongruents per a l'ajust de M1 ni per triar-ne hiperparàmetres si el protocol és estrictament de modelatge de la normalitat.
- Per S1, el grup és el text; per S2, `reference_id`; per S3, `tts_system`. Comprovar l'absència de grups compartits entre train/test per a cada protocol.
- La selecció de paràmetres de WavLM/SVM, el calibratge i qualsevol ajust de Ridge/covariància han de quedar **dins de la jerarquia d'entrenament**. Si una partició interna no és viable (p. ex. pocs TTS disponibles), fallar explícitament o documentar un esquema alternatiu aprovat: mai fer un CV agrupat invàlid silenciosament.
- Les prediccions Qwen per text es poden reutilitzar entre àudios amb el mateix text, però no s'ha d'ajustar Qwen ni calibrar-lo amb informació del test extern.

### Matemàtiques i consistència

- Probabilitats: 4 components finits, no negatius, suma aproximada 1; definir toleràncies i registrar correccions. No aplicar `softmax` a probabilitats ja normalitzades.
- B0: `1[argmax(P_A) != argmax(P_T)]`. Definir desempats (`argmax` amb l'ordre canònic) i reportar-ne la freqüència si es produeixen.
- B1: `JS(P_A,P_T) = 0.5 KL(P_A||M) + 0.5 KL(P_T||M)`, `M=(P_A+P_T)/2`, **base 2**, score entre 0 i 1. Tractar els zeros amb convencions numèricament segures (`0·log 0 = 0`).
- M1: Ridge multivariable (`P_T -> P_A esperat`), residual `r = P_A - P_A_esperat`, i score basat en la distància de Mahalanobis amb covariància regularitzada dels **residuals congruents**. Centrar segons la mitjana estimada dels residuals congruents; documentar la convenció exacta del score (`d` o `d²`).
- Les 4 probabilitats estan restringides a sumar 1. La covariància de 4 components pot ser singular: treballar en **3 coordenades independents** (o una transformació matemàticament defensable). No invertir una matriu singular directament. Si les prediccions Ridge surten del símplex, documentar i justificar com es tracten; no retallar ni renormalitzar silenciosament.
- **Mahalanobis no és una probabilitat calibrada de mismatch.** La interpretació com a `-log p(P_A|P_T, congruent)` requereix explicitar hipòtesis de densitat, constants i les coordenades utilitzades.

## Estructura nova recomanada (proposta, no estat actual)

```text
AGENTS.md
README.md
RESEARCH_PROTOCOL.md
src/emis_research/             # scripts originals, si és l'estructura efectiva
src/mismatch/
    __init__.py
    prepare_inputs.py
    b0_labels.py
    b1_js.py
    m1_conditional.py
    evaluate.py
    run_experiment.py
tests/
    test_mismatch_inputs.py
    test_b0.py
    test_b1_js.py
    test_m1.py
    test_splits_leakage.py
outputs/mismatch_v1/          # generats: no incloure'n contingut inventat
```

No crear tota l'estructura de cop si no és necessària. Mantenir l'estructura real del projecte i adaptar-hi els mòduls nous.

## Contracte mínim dels artefactes

Desar resultats per **`protocol`, `outer_fold`, `audio_id`, `role`** (`train_oof` o `test`) i versió de l'experiment. Les taules planejades són:

- `audio_predictions.csv`: etiquetes, quatre probabilitats acústiques, rol, identitat del model i informació de calibratge.
- `text_predictions.csv`: text/grup, quatre valors textuals, `option_mass`, identificació de Qwen i prompt.
- `mismatch_inputs.csv`: unió auditada amb metadades, etiqueta sintètica i partició.
- `mismatch_scores.csv`: B0, B1, M1 per àudio i fold; un valor absent s'ha de marcar com a no calculat, mai posar-lo a zero.
- `metrics_by_fold.csv`, `metrics_summary.csv`, `pairwise_results.csv`, `subgroup_results.csv`.
- `audit_report.json`: recompte de files, joins, classes, distribucions, cobertura i proves de no-fuga.

Aquests fitxers **no existeixen necessàriament**: són el contracte proposat, subjecte a revisió abans d'implementar-los.

## Tests mínims a exigir

1. B0: etiquetes iguals → 0; diferents → 1.
2. B1: distribucions idèntiques → 0; distribucions disjuntes one-hot → 1 (base 2); simetria; zeros vàlids; invariància a repetir càlcul.
3. Inputs: ordre de classes correcte; sumes aproximadament 1; 4 columnes; no valors NaN; join únic per `audio_id` i fold.
4. M1: shapes correctes, residuals calculats com a resta, covariància regularitzada invertible en 3 dimensions, score finit i no negatiu; cap entrenament amb `test`.
5. Splits: grups sense solapament, absència de `audio_id` duplicat en un mateix rol/fold, i reproduïbilitat de l'assignació.
6. Tests de **fuga d'informació** amb dades artificials: alterar etiquetes o valors de test no pot canviar els paràmetres apresos del fold.
7. Avaluació: emparellar scores i etiquetes per als mateixos `audio_id`; calcular correctament AUROC/AP fins i tot amb B0 discret; gestionar folds inviables explícitament.

## Què lliurar en acabar cada etapa

- Fitxers creats/modificats i justificació breu.
- Comanda exacta per reproduir el pas, **només si s'ha verificat** que funciona en l'estructura real.
- Tests executats i resultat (o limitació si falten dependències/fitxers).
- Artefactes generats amb rutes reals; no inventar-los.
- Decisions pendents, particularment calibratge, CV interna, estimació de la covariància i tractament del símplex.

## Prohibicions explícites

- No entrenar M1 amb exemples incongruents si s'està implementant la versió acordada.
- No escollir el millor mètode mirant el test i després presentar la mateixa avaluació com a independent.
- No comparar un B1 amb prediccions diferents de les emprades a M1 sense explicar-ho.
- No presentar 1.248 clips com 1.248 observacions textuals independents: hi ha 104 textos compartits.
- No descriure scores com a emocions reals percebudes per humans sense evidència.
- No incorporar M2, pèrdues contrastives ni altres arquitectures sense decisió científica explícita.
