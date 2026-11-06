# RESEARCH_PROTOCOL.md — Protocol científic B0 → B1 → M1

**Projecte:** SkillzU · detecció d'incongruència emocional acústica–semàntica en EMIS  
**Estat:** especificació metodològica prèvia a la implementació; no és un informe de resultats  
**Versió:** 1.0 · 2026-11-06  
**Abast:** B0 i B1 com a baselines; M1 com a model condicional. M2 es manté fora d'abast.

## 1. Pregunta científica i hipòtesi

### Pregunta de recerca

> **Estudiar si la modelització de l'emoció acústica esperada condicionada pel contingut emocional textual ofereix una millora discriminativa i de generalització respecte de mesures directes de desacord, utilitzant representacions unimodals congelades i un protocol experimental controlat sobre EMIS.**

Versió entenedora: **aprendre què és acústicament esperable per a un text ajuda a detectar millor la incongruència que comparar directament les prediccions de veu i text?**

### Hipòtesi contrastable

Un score construït amb una referència de congruència apresa (`M1`) pot capturar desviacions rellevants que el desacord categòric (`B0`) i la divergència directa de distribucions (`B1`) no modelen explícitament.

**Hipòtesi no demostrada.** És legítim concloure que M1 no millora B1 o que la millora no es manté en textos/TTS nous.

### Possible contribució

Una **avaluació controlada** de si la congruència condicional apresa millora la discriminació i la generalització en un conjunt sintètic amb protocols de grup. No presentar Ridge, JS, Mahalanobis ni el modelatge condicional com a algoritmes nous; tampoc afirmar novetat bibliogràfica sense una verificació específica de l'estat de l'art.

## 2. Què significa «mismatch» en aquest estudi?

Conservar tres conceptes clarament separats:

1. **Incongruència sintètica (objectiu experimental):** `y = int(text_emotion != generation_emotion)` segons el manifest EMIS.
2. **Desacord entre classificadors:** les branques acústica i textual donen prediccions diferents (B0 o B1).
3. **Incongruència percebuda:** una persona percep contradicció entre veu i significat. **No queda demostrada** per les metadades sintètiques.

La variable `y` del punt 1 és el *ground truth operatiu* per a aquesta comparació, **no una etiqueta humana gold perceptual**. Els textos neutrals poden presentar una emoció de generació diferent sense que això impliqui una contradicció perceptiva inequívoca; per això són un subgrup especialment important.

## 3. Corpus, unitat d'anàlisi i dependències

### Disseny verificat en scripts

`build_manifest.py` exigeix **1.248 àudios** i 12 àudios per grup de text (4 emocions de generació × 3 TTS). `make_splits.py` exigeix **104 grups de text**. La combinació implica 312 casos congruents i 936 incongruents si cada text té un exemple per emoció/TTS i l'emoció textual és una de les quatre classes.

Aquests totals són **esperats pel codi**, no verificats contra els àudios reals en l'estat actual del ZIP.

Camps importants: `audio_id`, `text`, `text_id`, `text_category`, `text_cue_type`, `text_emotion`, `generation_emotion`, `reference_id`, `tts_system`; `group_text = text_category + '_' + text_id` a `make_splits.py`.

- **Unitat de score:** un àudio (`audio_id`).
- **Unitat de dependència textual:** `group_text`. Els àudios amb un mateix text comparteixen semàntica i no són independents per a inferència estadística.
- **Família de classes:** `angry`, `happy`, `neutral`, `sad`, en aquest ordre canònic.
- **Tipus de pista semàntica:** `explicit`, `implicit`, `neutral`, quan estiguin disponibles al manifest.

## 4. Model unimodal i vectors comuns

Les representacions de les dues branques es mantenen **congelades conceptualment**: no es proposa entrenament conjunt d'extrem a extrem. Podrà ser necessari **ajustar el classificador SVM i el calibrador en cada fold**, tal com exigeix l'avaluació correcta.

### Branca acústica

`àudio → WavLM → embeddings → SVM-RBF → probabilitats calibrades`.

Definir, per a cada àudio i fold:

\[
P_A=(p^A_{angry},p^A_{happy},p^A_{neutral},p^A_{sad})
\]

Els embeddings WavLM poden tenir centenars/milers de dimensions. **M1 no els rep:** rep només les **4 probabilitats emocionals** del classificador acústic.

**Gap tècnic real:** l'script `audio_wavlm_predict.py` inspeccionat fa `.predict()` en la CV externa; no exporta el `P_A` calibrat per registre ni les prediccions OOF apropiades per entrenar Ridge. Cal completar aquesta peça abans d'executar B1/M1 amb rigor.

### Branca textual

`text → Qwen → puntuacions per emoció → normalització entre les quatre opcions`.

\[
P_T=(p^T_{angry},p^T_{happy},p^T_{neutral},p^T_{sad})
\]

La normalització Qwen **no assegura probabilitats calibrades**; guardar `option_mass` i registrar aquesta limitació. No ajustar cap calibratge amb el test extern. Les prediccions de text són comunes als àudios amb el mateix `group_text`; fer unions verificades, no duplicacions ambigües.

### Validacions de tots dos vectors

- 4 components en ordre canònic, valors finits i dins `[0,1]` amb tolerància numèrica justificada.
- Suma per fila propera a 1; registrar qualsevol excepció.
- Associació inequívoca entre `audio_id`, text, fold, predicció i versió de model.
- No barrejar puntuacions crues, logits, probabilitats calibrades i distribucions normalitzades com si fossin equivalents.

## 5. Definició dels mètodes

### B0 — Desacord d'etiquetes

\[
\widehat y_A=\arg\max_c P_A(c),\qquad
\widehat y_T=\arg\max_c P_T(c)
\]
\[
B0=\mathbb{1}[\widehat y_A\neq\widehat y_T]
\]

Un score **binari** (0 o 1). Defineix explícitament el tractament dels empats mitjançant l'ordre de classes establert. B0 es pot calcular amb etiquetes de sortida sense esperar el calibratge, però per a una comparació final homogènia convé derivar-lo dels mateixos vectors que B1 i M1. Registrar si el calibratge canvia etiquetes o ranking de classes.

### B1 — Divergència Jensen–Shannon

\[
M=\tfrac12(P_A+P_T)
\]
\[
B1=JS(P_A,P_T)=\tfrac12 KL_2(P_A\parallel M)+\tfrac12 KL_2(P_T\parallel M)
\]

On `KL_2` utilitza logaritme en base 2. Resultat teòric entre **0 i 1** per dues distribucions discretes. Un valor gran indica més divergència, no una probabilitat de mismatch. Definir el comportament numèric per a zeros sense introduir un epsilon que alteri injustificadament els extrems.

**JS no s'entrena.** Comparar B1 contra M1 sempre amb les mateixes versions de `P_A` i `P_T`.

### M1 — Referència de congruència condicional apresa

**Objectiu:** modelar què és acústicament esperable a partir del text **quan el parell és congruent**, i després puntuar la desviació de la veu real.

**Pas A: ajustar Ridge sobre l'entrenament congruent.**

Per a cada mostra congruent d'entrenament, emprar `X=P_T` i `Y=P_A` **OOF** (predicció acústica obtinguda sense entrenar el classificador sobre aquell exemple). Ajustar una regressió multivariable amb regularització L2:

\[
\widehat P_A(P_T)=WP_T+b
\]

L'hiperparàmetre Ridge (`alpha`) s'ha de fixar *a priori* o escollir només dins de les dades d'entrenament amb una estratègia de validació compatible amb els grups; **no** seleccionar-lo segons el resultat del test ni segons la millor AUROC final del test.

**Pas B: residuals congruents i variabilitat habitual.**

\[
r_i=P_{A,i}-\widehat P_A(P_{T,i})
\]

Amb els residuals de referència de l'entrenament congruent, estimar la seva mitjana `mu_r` i una covariància regularitzada `Sigma_reg` (p. ex. Ledoit–Wolf). **Cal documentar com s'han generat aquests residuals de referència i evitar que una avaluació sobre els mateixos exemples de Ridge n'infravalori la variància sense control.** Si es requereix estimació de residuals fora de mostra de Ridge, fer-ne *cross-fitting* dins de l'entrenament extern.

Les probabilitats estan sobre un símplex de dimensió 3. Per estimar la covariància, emprar **3 coordenades independents** o una transformació log-ratio justificada. La versió inicial simple pot retenir els primers tres components en ordre canònic si els vectors predit i observat compleixen consistentment la restricció de suma 1. Si Ridge genera prediccions fora del símplex, registrar-ho i definir explícitament el tractament abans de calcular scores; no projectar automàticament sense documentar-ne l'efecte.

**Pas C: puntuació del test.**

Per a una mostra de test, predir `P_A_esperat` a partir de `P_T`; calcular `r_test = P_A_test - P_A_esperat` i puntuar:

\[
M1=(r_{test}-\mu_r)^\top\Sigma_{reg}^{-1}(r_{test}-\mu_r)
\]

Aquesta és la **distància de Mahalanobis al quadrat**. Un valor alt vol dir que la desviació és poc habitual comparada amb els residuals congruents d'entrenament.

- El model **no utilitza les etiquetes `generation_emotion` com a objectiu Ridge**: són necessàries per definir els exemples congruents. El target numèric Ridge és el vector `P_A`.
- El M1 acordat **no és** un classificador supervisat entrenat amb congruents i incongruents.
- El score M1 **no és directament** `P(mismatch)`. Una interpretació com a probabilitat o `−log p(P_A|P_T, congruent)` requeriria un model de densitat normalitzat i hipòtesis addicionals.
- Si la distribució dels residuals és molt no gaussiana, això limita la interpretació probabilística de Mahalanobis; s'ha de registrar i, si és rellevant, estudiar posteriorment.

## 6. Divisió experimental i regles contra el *data leakage*

### Protocols fixats al codi original

| Protocol | Separació externa | Què permet afirmar? |
|---|---|---|
| **S1** | 5 folds amb `StratifiedGroupKFold`, agrupats per `group_text`; llavor 42 | Rendiment sobre **textos no vistos** dins de la resta de condicions del corpus |
| **S2** | Un `reference_id` fora per fold | Rendiment sobre **referències no vistes** |
| **S3** | Un `tts_system` fora per fold | Rendiment sobre **TTS no vistos**, dins dels tres TTS del corpus |

Un fold S3 entrena amb només **dos** dels tres sistemes TTS: una CV interna que exigeixi més grups TTS no és admissible; cal anticipar aquesta limitació i aprovar-ne el tractament abans d'executar.

### Esquema obligatori per cada fold extern

```text
1. Llegir la separació externa fixada per make_splits.py.
2. Reservar completament el test extern.
3. Dins de train extern:
   a. Seleccionar WavLM/SVM i hiperparàmetres amb CV interna agrupada vàlida.
   b. Ajustar calibratge acústic sense usar el test extern.
   c. Produir P_A train OOF de la cadena de predicció completa per a Ridge.
   d. Ajustar Ridge exclusivament sobre congruents d'entrenament.
   e. Estimar residuals i covariància regularitzada només des de congruents d'entrenament.
   f. Fixar hiperparàmetres/llindars permesos només amb entrenament.
4. Ajustar la cadena acústica final amb les dades permeses del train extern.
5. Produir P_A i P_T del test, i scores B0/B1/M1 sense reajustar res al test.
6. Guardar les prediccions, els identificadors i les configuracions del fold.
```

**Atenció important:** `train OOF` i `test` poden procedir d'ajustos diferents de l'SVM. Això pot canviar-ne la distribució de confiança; cal auditar-la i assegurar una política coherent de calibratge/ajust. Els detalls exactes de *cross-fitting* (inclòs Ridge, si escau) s'han de decidir abans de l'execució, no improvisar després de mirar els resultats.

La dependència per text afecta també l'estimació de la variabilitat i la precisió estadística: la unitat de resampling inferencial principal serà el **text**, no cadascun dels dotze clips com si fossin independents.

## 7. Avaluació i comparacions

### Objectiu principal

**S1, AUROC de detecció d'incongruència per fold extern.** Comparar M1 contra B1 (comparació principal) i contra B0 (baseline elemental), sobre els mateixos àudios de cada fold.

### Mètriques secundàries

- **Average Precision (AP/PR-AUC)**: important per la proporció de classes; informar del baseline de prevalença.
- **Balanced accuracy i F1**: només amb llindars escollits mitjançant dades d'entrenament, mai test.
- **Resultats per fold** i agregats, sense amagar la variabilitat.
- **Avaluació parell a parell** sobre variants del mateix text sota diferents emocions de generació, quan sigui pertinent i clarament definida; declarar la unitat estadística i el tractament de parelles dependents.
- **S2 i S3**: robustesa fora de referència/TTS, amb limitacions pel baix nombre de grups.
- **Subgrups**: `explicit`, `implicit`, `neutral` i altres subgrups presents i amb mida suficient; evitar afirmar conclusions fortes sobre subgrups petits.

Una estratègia de bootstrap o d'incertesa ha de **remostrejar grups textuals sencers**, especialment per a S1. Comparar diferències aparellades M1–B1; no calcular intervals tractant tots els clips com a independents.

### Baselines i desequilibri

Si el conjunt efectiu manté 75% d'incongruents, el classificador trivial «tot incongruent» ja obtindria **75% d'accuracy**. Per això no prendre accuracy global com a indicador principal. B0 és un score discret: l'AUROC es pot calcular, però té molts empats i menor resolució de ranking que B1/M1.

### Resultats que donarien suport o refutarien la hipòtesi

- **Suport:** M1 supera B1 en dades de test extern, la diferència és robusta als grups de text i la tendència es manté en protocols/subgrups pertinents.
- **Resultat negatiu vàlid:** M1 no millora B1; o només millora S1 però falla en TTS nous; o l'efecte desapareix per tipus de text.
- **Interpretació limitada:** una millora de la detecció de l'etiqueta sintètica EMIS no demostra automàticament millor predicció de judicis humans.

## 8. Artefactes, auditories i reproduïbilitat

Proposta de sortides sota `outputs/mismatch_v1/` (encara no creades):

| Artefacte | Contingut mínim |
|---|---|
| `audio_predictions.csv` | `protocol`, `outer_fold`, `audio_id`, `role`, `P_A` de 4 classes, predicció, identificador d'ajust i calibratge |
| `text_predictions.csv` | identificador de text, `P_T` de 4 classes, `option_mass`, model/prompt/versió |
| `mismatch_inputs.csv` | unió de manifest, splits, prediccions i `y_mismatch`, amb cobertura auditada |
| `mismatch_scores.csv` | `protocol`, `outer_fold`, `audio_id`, B0, B1 i M1, versions i rol |
| `metrics_by_fold.csv` | AUROC, AP i mètriques secundàries per mètode i fold |
| `metrics_summary.csv` | resum de mètriques i dispersió, sense barrejar protocols |
| `pairwise_results.csv` | comparacions parell a parell i identificadors dels grups |
| `subgroup_results.csv` | resultats per tipus de pista textual |
| `audit_report.json` | recomptes, duplicats, errors d'unió, qualitat de probabilitats, proves de no-leakage, versions |

Guardar també una **configuració de cada ajust**, els paràmetres Ridge i de covariància, la llavor, el hash de splits/manifest, la versió del codi i de les llibreries. Els fitxers no disponibles s'han de marcar com a **pendents**, no substituir-los per valors ficticis.

### Comprovacions abans de calcular mètriques

- Cobertura completa dels `audio_id` de cada test esperat; `join` un-a-un on correspongui.
- Correspondència exacta de classes i de `P_A`/`P_T`.
- Cap `test` en cap procés d'entrenament, calibratge, selecció ni covariància.
- M1 entrenat només amb `y_mismatch = 0` **de train**.
- AUROC/AP definibles (presència de positius i negatius); si no, registrar `NA` i la causa.
- Comparació aparellada B0/B1/M1 sobre la mateixa població avaluada.

## 9. Pla d'implementació amb punts de control

### Fase 0 — Auditoria de l'entorn (sense entrenar)

Inspeccionar la ubicació real de scripts, àudios, manifest, splits, característiques, prediccions i dependències. Confirmar la compatibilitat de les rutes. Verificar que les versions del corpus compleixen les assercions o documentar discrepàncies.

**Criteri de sortida:** fonts localitzades i contracte de dades aprovat.

### Fase 1 — Baseline B0

Preparar les prediccions categòriques per àudio i executar B0 amb tests simples. Confirmar que `text_emotion` i `generation_emotion` es fan servir exclusivament per a la veritat sintètica, no per substituir les prediccions de les branques.

**Criteri de sortida:** B0 reproduïble, amb un score per àudio de test i grup/split correcte.

### Fase 2 — Probabilitats i B1

Dissenyar i implementar la calibració acústica agrupada sense fuites, exportar `P_A` per àudio, validar les puntuacions Qwen `P_T`, implementar JS amb proves matemàtiques.

**Criteri de sortida:** B0/B1 calculats amb entrades homologables, tests de sumes/rangs/simetria superats.

### Fase 3 — M1

Construir `P_A train_oof` apropiat, ajustar Ridge només amb congruents d'entrenament, estimar residuals i covariància regularitzada (3 coordenades), calcular el score Mahalanobis de test. Auditar l'efecte de la restricció de suma 1 i la possible diferència entre sortides OOF i model final.

**Criteri de sortida:** score M1 reproduïble i sense fugues, amb paràmetres i tractament matemàtic documentats.

### Fase 4 — Avaluació i discussió

Calcular mètriques per fold, diferències aparellades i subgrups; comprovar robustesa S2/S3 on la configuració ho permeti. Distingir clarament troballes, hipòtesis i limitacions.

**Criteri de sortida:** taules per fold, agregats, auditoria i conclusions proporcionals a les dades.

## 10. Decisions específiques que cal tancar abans de programar M1

Aquestes decisions **encara no es poden donar per resoltes** pel fet d'haver escollit Ridge + Mahalanobis:

1. **Mètode exacte de calibratge de l'SVM** i particions internes viables per S1/S2/S3, especialment amb pocs grups de S3.
2. **Com construir OOF per entrenar Ridge** sense contaminació per selecció d'hiperparàmetres, i com harmonitzar OOF i test.
3. **Com generar residuals congruents de referència:** prediccions sobre els exemples que ajusten Ridge versus *cross-fitting* intern; avaluar-ne el biaix.
4. **Regularització Ridge (`alpha`)** i estimador/regularització de covariància: valors predefinits o validació agrupada dins de l'entrenament.
5. **Restricció de probabilitats** de l'output Ridge i les 3 coordenades independents: definir convenció estable, incloent el cas de prediccions fora del símplex.
6. **Llindars per balanced accuracy/F1**, seleccionats només a l'entrenament. Els scores B0/B1/M1 no tenen automàticament un llindar universal equivalent.
7. **Anàlisi d'incertesa i jerarquia d'agregació**: evitar pseudoreplicació per textos i interpretacions fortes a S3 amb tres TTS.

Es poden implementar B0 i els tests de JS mentre es tanquen aquests detalls, però **no executar un M1 definitiu ni comunicar-ne resultats** fins que les decisions necessàries quedin registrades.

## 11. Límits de validesa

- **EMIS és sintètic:** els resultats no acrediten automàticament generalització a parla espontània real.
- **Etiquetes de generació ≠ emoció percebuda:** no reclamar detecció perceptiva gold sense estudi humà.
- **Desacord entre models ≠ incongruència real:** errors i calibratge de WavLM/SVM/Qwen poden produir o ocultar desacords.
- **Generalització limitada:** S1 prova textos nous dins del corpus; S3 només tres TTS coneguts en la comparació total.
- **Pocs textos independents:** 104 grups de text; evitar models massa flexibles i inferència basada en clips independents.
- **Novelty:** no reivindicar novetat absoluta del modelatge probabilístic d'incongruència; el valor possible és un resultat empíric concret i reproduïble.

## 12. Documents relacionats

- [`README.md`](README.md) — resum explicatiu i estat del codi.
- [`AGENTS.md`](AGENTS.md) — instruccions operatives i restriccions per a Codex.

**Regla final:** l'experiment ha de poder concloure legítimament «M1 no millora B1». Una metodologia que impedeixi arribar a aquesta conclusió no és adequada per respondre la pregunta científica.
