# Anàlisi exploratòria S1 de B0, B1 i M1

Script: `src/emis_research/mismatch/explore_subgroups.py`. Només llegeix
scores de test extern, manifest i splits. No calcula probabilitats ni ajusta
models. Claus exactes: audio_id, protocol, outer_fold; rol test, unicitat,
etiquetes i metadades contrastades amb el manifest. 1.248 àudios, 104 textos.
Els literals text_cue_type són explicit, implicit i neutral: 39, 39 i 26
textos, respectivament. No es recodifiquen categories.

S'exporten AUROC/AP pooled per subgrup i diferències M1-B1 i B1-B0.
Bootstrap aparellat M1-B1 amb 10.000 rèpliques per subgrup, seed=42, tots els
textos del subgrup remostrejats amb reemplaçament i tots els seus clips
repetits segons la multiplicitat. IC percentils 2.5/97.5. No s'imputen ni
substitueixen rèpliques sense les dues classes. El codi omet l'interval amb
menys de dos grups o una sola classe; això és viabilitat computacional,
no una garantia de precisió inferencial. Es mostren sempre mides i validesa.

Percentils descriptius globals dels 1.248 scores: rang mitjà/N × 100,
tractant explícitament els empats. No són probabilitats ni llindars.
Exemples: els tres textos amb més diferència de percentils a favor de cada
mètode i els tres amb major mínim dels dos percentils (coincidència alta).
Un àudio per text i categoria; desempats per audio_id. Són exemples extrems
seleccionats per una regla reproduïble, no una mostra representativa del corpus.

Parelles: cada variant sintèticament incongruent es compara amb la congruent
del mateix group_text i tts_system. Hi ha 9 parelles per text i 936 totals.
S'exporten victòries estrictes, empats i derrotes; concordança = 1, 0.5 o 0.
El resum fa mitjana per text, sense tractar parelles com a independents.
Les referències acústiques no s'igualen artificialment: es registra si són
iguals. Aquesta anàlisi controla text i TTS, però pot variar la referència.

El CSV M1 va serialitzar reference_id sense zeros inicials. Es comprova
equivalència numèrica i s'utilitza la forma original del manifest a les
noves taules. No es canvien dades ni resultats dels experiments previs.

Sortides noves amb prefix `explore_s1_`: subgroup_metrics.csv,
subgroup_bootstrap_summary.csv, subgroup_bootstrap_replicas.csv,
relative_scores.csv, examples.csv, within_text_pairs.csv,
within_text_summary.csv i audit.json. Mode de creació exclusiu i verificació
de hashes de tots els artefactes previs abans d'exportar.

Execució des de l'arrel:

```powershell
.\.venv\Scripts\python.exe -B src/emis_research/mismatch/explore_subgroups.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_explore_subgroups.py -v
```

Interpretació exploratòria: no hi ha correcció per comparacions múltiples,
ni selecció posterior de models/subgrups. El bootstrap manté fixos els models
i no estima variabilitat de reentrenament. Els 26–39 textos de cada subgrup
limiten la precisió. B0 és preliminar sense calibratge acústic, mentre que
B1 i M1 comparteixen vectors calibrats. Els resultats descriuen discriminació
de condicions sintètiques de generació EMIS, sense validació perceptual humana.
