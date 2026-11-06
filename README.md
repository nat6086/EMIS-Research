# SkillzU — Incongruència emocional acústica–semàntica (EMIS)

## 1. De què tracta el projecte?

Aquest projecte estudia si podem detectar quan **l'emoció expressada per una veu** no encaixa amb **l'emoció associada al text pronunciat**.

Ho estudiem amb **EMIS**, un corpus sintètic en què les frases i les condicions emocionals de generació dels àudios estan identificades. En aquest projecte, la variable objectiu és **incongruència de generació** (`text_emotion != generation_emotion`), que **no equival necessàriament** a una contradicció emocional percebuda per humans.

La pregunta científica és:

> **La modelització de l'emoció acústica esperada condicionada pel contingut emocional textual permet millorar la discriminació i la generalització de la incongruència emocional respecte de comparacions directes entre les prediccions de les dues modalitats?**

La resposta s'obtindrà experimentalment. **Encara no s'ha demostrat que el model condicional sigui superior.**

## 2. Les dues branques del sistema

### Branca acústica (veu)

```text
Àudio → WavLM (característiques grans) → SVM-RBF → calibratge → P_A (4 valors)
```

WavLM obté característiques acústiques. Un classificador SVM-RBF prediu l'emoció de la veu; per als experiments B1 i M1 cal disposar de **probabilitats calibrades per les quatre emocions**, no només de l'etiqueta guanyadora.

### Branca semàntica (text)

```text
Text → Qwen → P_T (4 valors)
```

Qwen assigna puntuacions a les quatre categories emocionals i el script existent les normalitza. **Una normalització no és sinònim de calibratge**; aquesta limitació s'ha de documentar i, si s'estima necessari, estudiar dins de les particions d'entrenament.

**Ordre de les classes a tot el projecte:** `angry`, `happy`, `neutral`, `sad`.

## 3. Els tres mètodes que compararem

| Identificador | Nom | Idea explicada fàcilment |
|---|---|---|
| **B0** | Desacord d'etiquetes | «La veu i el text prediuen emocions diferents?» Retorna 0 o 1. |
| **B1** | Jensen–Shannon (JS) | «Com de diferents són els vectors de probabilitats de veu i text?» |
| **M1** | Model de congruència condicional | «Com d'inesperada és la veu real respecte del que esperaríem per a aquest text?» |

### B0 — Etiquetes

Compara l'etiqueta dominant de `P_A` amb la de `P_T`.

```text
B0 = 0, si coincideixen; B0 = 1, si no coincideixen.
```

### B1 — Jensen–Shannon

Compara directament **els dos vectors de quatre probabilitats**. Amb logaritme en base 2, retorna un score entre 0 (iguals) i 1 (totalment divergents).

### M1 — Ridge + Mahalanobis

M1 afegeix un **model petit** que s'entrena **únicament amb els exemples congruents del conjunt d'entrenament**:

```text
Entrenament:
    P_T (Qwen) ──> Ridge ──> P_A esperat
                    ↑
        aprèn amb P_A obtingut fora de mostra
        d'àudios congruents de l'entrenament

Predicció:
    P_T ──> Ridge ──> P_A esperat
                              ┐
    Àudio ──> SVM ──> P_A real ┴──> residuals ──> Mahalanobis ──> score M1
```

- **Ridge** aprèn les probabilitats acústiques esperades a partir de les textuals.
- **Residual** = vector acústic real menys vector esperat.
- **Mahalanobis** puntua fins a quin punt els residuals són atípics respecte de la variabilitat habitual dels exemples congruents.

**Ridge només prediu quatre valors emocionals**, no els embeddings extensos de WavLM. Per al càlcul de la covariància podem usar tres coordenades independents, perquè les quatre probabilitats sumen 1.

Mahalanobis produeix un **score d'atipicitat**, no directament una probabilitat calibrada de mismatch.

## 4. Quines dades tenim previstes?

D'acord amb els scripts del manifest i dels splits:

| Propietat | Disseny previst |
|---|---:|
| Textos diferents | 104 |
| Emocions de generació per text | 4 |
| Sistemes TTS | 3 |
| Àudios totals | 1.248 |
| Àudios congruents (`text_emotion == generation_emotion`) | 312 |
| Àudios incongruents | 936 |

**Aquests són recomptes esperats segons el disseny i les assercions del codi, no recomptes reproduïts ara a partir dels fitxers d'àudio.** Els àudios i el CSV de frases no eren al ZIP de codi revisat.

La desigualtat 25% congruents / 75% incongruents implica que l'**accuracy global sola pot ser enganyosa**. Utilitzarem mètriques de discriminació més informatives.

## 5. Protocols d'avaluació

Els scripts defineixen tres maneres de separar entrenament i test:

| Protocol | Què queda fora de l'entrenament? | Què posa a prova? |
|---|---|---|
| **S1** | Textos sencers, 5 folds | Generalització a textos nous |
| **S2** | Identificadors `reference_id` | Generalització a referències acústiques no vistes |
| **S3** | Sistemes `tts_system` | Generalització a sistemes TTS no vistos |

**S1 és l'anàlisi principal prevista**; S2 i S3 ofereixen proves complementàries de robustesa. Cal respectar les dependències entre mostres: dotze àudios comparteixen un mateix text.

Per evitar fuites d'informació (*data leakage*), la selecció de models, el calibratge, Ridge i la covariància s'han d'ajustar **sense veure el test extern**. El protocol complet consta a [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md).

## 6. Què existeix al codi de partida?

El ZIP `emis_research(1).zip` inspeccionat conté scripts Python amb els noms:

```text
build_manifest.py
make_splits.py
audio_wavlm_extract.py
audio_wavlm_predict.py
text_emotion_extract_predict.py
text_emotion_evaluate.py
audio_egemaps_extract.py
audio_egemaps_predict.py
audio_signal_extract.py
audio_signal_predict.py
```

- **Manifest**: `build_manifest.py` preveu generar `data/processed/emis_manifest.csv`.
- **Splits**: `make_splits.py` preveu generar `configs/splits/splits_v1.csv` i un JSON de traçabilitat.
- **Acústica**: `audio_wavlm_predict.py` ja incorpora CV aniuada amb agrupacions i avalua diverses famílies, incloent SVM-RBF; actualment empra `.predict()` i no proporciona, tal com caldrà per M1, un export complet de probabilitats calibrades i prediccions OOF d'entrenament.
- **Text**: `text_emotion_extract_predict.py` inclou una opció basada en Qwen i registra `option_mass`.

**Estat verificat:** el ZIP conté codi, però **no conté els fitxers d'àudio, els artefactes processats ni els resultats dels nous experiments B0/B1/M1**. Els rendiments acústics o textuals esmentats durant la recerca no s'han reproduït amb aquest ZIP.

**Ubicació dels scripts:** dins del ZIP els fitxers es troben a `emis_research/`, mentre que els comentaris i els càlculs de rutes del codi pressuposen una disposició de tipus `src/emis_research/`. Quan s'integri al repositori de VS Code, cal verificar la ubicació real abans d'executar scripts.

## 7. Estructura de treball proposada

```text
projecte/
├── AGENTS.md                 # regles per a Codex
├── README.md                 # aquest document
├── RESEARCH_PROTOCOL.md      # definicions i disseny científic
├── src/
│   ├── emis_research/        # scripts originals, si aquesta és la ruta efectiva
│   └── mismatch/             # mòduls futurs B0, B1, M1 i avaluació
├── tests/                    # proves automàtiques futures
├── data/                     # dades locals (si disponibles)
├── configs/splits/           # splits generats
└── outputs/mismatch_v1/      # resultats generats, encara no creats
```

Els mòduls proposats a `src/mismatch/` són `prepare_inputs.py`, `b0_labels.py`, `b1_js.py`, `m1_conditional.py`, `evaluate.py` i `run_experiment.py`. **Són una proposta d'organització, no fitxers ja implementats.**

## 8. Ordre de desenvolupament

1. **Auditoria d'entrades:** manifest, splits, identificadors, etiqueta sintètica, cobertura i probabilitats disponibles.
2. **B0:** càlcul i tests del desacord d'etiquetes.
3. **Probabilitats acústiques i B1:** calibratge sense fuites, export de prediccions, JS i tests.
4. **M1:** OOF acústic adequat, Ridge amb congruents, covariància regularitzada, Mahalanobis i tests.
5. **Avaluació:** AUROC i altres mètriques per fold, comparació aparellada, anàlisi per subgrups i robustesa S2/S3.

No començar per M1 sense haver comprovat que `P_A` i `P_T` estan alineats, normalitzats i construïts correctament per a cada fold.

## 9. Què voldrem obtenir al final?

- Resultats B0, B1 i M1 **sobre exactament els mateixos exemples de test**.
- Comparació d'AUROC per folds S1 i mètriques complementàries (AP, balanced accuracy, F1 segons llindars triats només amb entrenament).
- Resultats S2/S3 per examinar robustesa.
- Anàlisi de textos `explicit`, `implicit` i `neutral`, distingint l'etiqueta sintètica de la percepció humana.
- Intervals d'incertesa o contrastos que tinguin en compte els **grups de text**, no clips tractats com a independents.
- Artefactes traçables: probabilitats per àudio, scores, mètriques, configuracions i auditoria.

## 10. Com treballar amb Codex a VS Code

Obre **la carpeta arrel real del projecte**, amb aquests tres documents a l'arrel. Dona a Codex una tasca petita cada vegada, per exemple:

> Llegeix `AGENTS.md` i `RESEARCH_PROTOCOL.md`. Inspecciona l'estructura real del repositori i els scripts existents. Sense modificar res encara, digues quins fitxers i quines sortides falten per poder implementar B0 sense data leakage.

Un cop revisat el diagnòstic, passa a la implementació en etapes. Evita instruccions massa obertes com «implementa tot el sistema» sense punts de verificació.

## 11. Documents de referència

- [`AGENTS.md`](AGENTS.md): instruccions de treball per a l'assistent de codi.
- [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md): metodologia científica, scores, avaluació i restriccions de validesa.

**Estat del projecte:** documents de planificació preparats; implementació de B0/B1/M1 i resultats encara pendents.


## 12. Estructura del projecte

El projecte segueix una estructura modular que separa les dades, el codi font, les configuracions, els models i els resultats experimentals.

```text
emis_research/
│
├── AGENTS.md                  # Instruccions per a Codex
├── README.md                  # Documentació general del projecte
│
├── configs/                   # Configuracions dels experiments
│
├── data/
│   ├── raw/                   # Dataset original, sense modificar
│   ├── interim/               # Dades intermèdies
│   └── processed/             # Dades processades
│
├── src/
│   └── emis_research/         # Codi Python de la pipeline
│       ├── ...                # Scripts existents de WavLM i Qwen
│       └── mismatch/          # Mòdul nou (proposat)
│           ├── prepare_inputs.py
│           ├── b0_labels.py
│           ├── b1_js.py
│           ├── m1_conditional.py
│           ├── evaluate.py
│           └── run_experiment.py
│
├── tests/                     # Tests automàtics
├── notebooks/                 # Exploració i anàlisi
│
├── models/
│   └── checkpoints/           # Pesos dels models
│
├── outputs/
│   └── mismatch_v1/           # Resultats dels experiments B0/B1/M1
│
└── docs/
    └── RESEARCH_PROTOCOL.md   # Pregunta científica i metodologia
```

### 12.1. Organització del codi

El codi principal es troba a `src/emis_research/`, que conté la pipeline existent d'extracció de característiques acústiques, classificació emocional i predicció semàntica.

Es proposa incorporar un submòdul `mismatch/` per implementar tres mètodes de detecció d'incongruència acústica-semàntica:

- **B0 — Desacord d'etiquetes:** compara les etiquetes emocionals predites per les dues branques.
- **B1 — Jensen–Shannon:** calcula la divergència entre les distribucions de probabilitat emocional.
- **M1 — Conditional Surprise:** aprèn l'emoció acústica esperada condicionada pel text i calcula un score d'atipicitat mitjançant els residuals i la distància de Mahalanobis.

### 12.2. Resultats experimentals

Els resultats de B0, B1 i M1 es desaran a `outputs/mismatch_v1/`, mantenint-los separats de les dades originals i dels resultats previs.

Les configuracions dels experiments es gestionaran des de `configs/`, mentre que els tests es trobaran a `tests/`.

### 12.3. Estat de la implementació

El directori `mismatch/` i els seus scripts representen **l'arquitectura prevista** per a la nova fase experimental. No s'han d'interpretar com a components ja implementats.

Es conservarà l'estructura original del projecte i s'evitaran modificacions innecessàries a la pipeline existent.