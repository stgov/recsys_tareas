# Tarea 1: recomendación de recetas

Este directorio reúne el EDA, los experimentos y la evidencia usada por
[`INFORME_BORRADOR.md`](INFORME_BORRADOR.md). El enunciado está en
[`Tarea_1_RecSys_2026_2.pdf`](../Tarea_1_RecSys_2026_2.pdf).

Los notebooks originales del equipo se conservan en `notebooks/`; no se crearon
copias alternativas. `utils.py` tampoco se modifica: la evaluación auditable se
encapsula en `evaluation.py` para no cambiar la API compartida.

## Estructura

- `notebooks/eda.ipynb` y `metadata_eda.ipynb`: calidad, cobertura, dispersión,
  temporalidad y metadata.
- `notebooks/user_based.ipynb`, `item_based.ipynb` y `funksvd.ipynb`: grillas,
  selección y evaluación de cada familia.
- `notebooks/general.ipynb`: comparación común de todos los métodos obligatorios.
- `evaluation.py`: candidatos, métricas, diagnósticos, desempate y bootstrap.
- `ranking_models.py`: adaptadores para ALS/BPR implícitos y Content-Based.
- `experiments_item_funksvd.py`: única implementación experimental de ItemKNN
  y FunkSVD usada por los notebooks y la línea de comandos.
- `general_experiments.py`: UserKNN, baselines, ALS, BPR, Content-Based y consolidación.
- `report_eda.py` y `plot_item_funksvd_results.py`: tablas y figuras del informe.
- `informe_assets/`: figuras PNG/PDF y evidencia compacta versionable.

`dataset/`, `results/` y `figures/` son ubicaciones locales o históricas
ignoradas por Git. Las conclusiones nuevas deben usar solamente
`informe_assets/`.

## Entorno y ejecución completa

Desde la raíz de `recsys_tareas`:

```powershell
uv sync --dev
uv run python -m tarea_01.report_eda

uv run jupyter nbconvert --to notebook --execute --inplace `
  tarea_01/notebooks/eda.ipynb tarea_01/notebooks/metadata_eda.ipynb `
  --ExecutePreprocessor.timeout=1800

uv run jupyter nbconvert --to notebook --execute --inplace `
  tarea_01/notebooks/user_based.ipynb tarea_01/notebooks/item_based.ipynb `
  tarea_01/notebooks/funksvd.ipynb --ExecutePreprocessor.timeout=1800

uv run jupyter nbconvert --to notebook --execute --inplace `
  tarea_01/notebooks/general.ipynb --ExecutePreprocessor.timeout=2400

uv run python -m tarea_01.plot_item_funksvd_results
uv run python -m unittest discover -s tarea_01/tests -v
```

El orden importa: `general.ipynb` lee las configuraciones seleccionadas por los
tres notebooks de filtrado colaborativo.

ItemKNN y FunkSVD también pueden ejecutarse sin Jupyter:

```powershell
uv run python -m tarea_01.experiments_item_funksvd
```

La grilla predeterminada compara cuatro vecindarios y dos similitudes para
ItemKNN, y tres dimensiones, dos tasas de aprendizaje y dos regularizaciones
para FunkSVD. Para una prueba rápida se puede reducir la grilla y las épocas.
`--skip-ranking` calcula solo RMSE/MAE. `--ranking-negatives full` usa todo el
catálogo no visto y puede ser considerablemente más lento.

## Protocolo común

- Entrenamiento: todas las 157.108 filas de `train.csv`.
- Evaluación de ratings: las 27.416 observaciones reales de `validation.csv`.
- Ranking: todos los ítems de validación del usuario, incluso los no relevantes,
  más hasta 100 ítems no observados del catálogo de entrenamiento.
- Relevancia: `rating >= 4`; Top-K: `k=10`; semilla: `42`.
- Los negativos no incluyen ítems vistos en train ni otros ítems observados en
  validación. Son no observados, no rechazos conocidos.
- Todos los modelos usan candidatos idénticos, ordenados de forma estable. La
  corrida actual registra el hash
  `22d8fdfe9f87ae72b93c3379f56f6fb6738babcc4df95b61b76e58982a404743`.
- El ranking usa scores crudos (`clip=False`). Los empates se resuelven por
  `recipe_id` ascendente, sin consultar la etiqueta. RMSE/MAE usan únicamente
  ratings reales y predicciones recortadas a la escala `[0, 5]`.
- Precision, Recall, nDCG y MAP se promedian sobre usuarios con al menos un ítem
  relevante. Diversidad, novedad y cobertura usan todos los usuarios evaluados.

Los resultados son de desarrollo: los hiperparámetros se eligen y se informan
sobre `validation.csv`; no existe un test independiente. Las métricas de ranking
son condicionales al muestreo de candidatos y no deben describirse como ranking
sobre catálogo completo.

## Métricas y nombres

- `RMSE` y `MAE`: error de predicción explícita.
- `Precision@10`, `Recall@10`, `NDCG@10`, `MAP@10`: calidad del ranking.
- `TagDiversity`: cantidad media de tags distintos en cada lista. Es un proxy
  amplio, no una taxonomía validada de estilos culinarios.
- `InteractionILD`: disimilitud coseno media entre vectores binarios de
  interacción de los ítems recomendados. No debe rotularse como tags o estilos.
- `NoveltyBitsPerItem`: media de `-log2 p(i)` por recomendación.
- `NoveltyBitsPerList`: suma de esa información por lista, promediada entre
  usuarios; corresponde a la escala solicitada por el enunciado.
- `CatalogCoverage`: fracción del catálogo de train recomendada al menos una vez,
  condicionada al protocolo de candidatos.
- `ActualK*`, `RankingFallbackFraction` y `RankingTieAt10Fraction`: diagnósticos
  necesarios para interpretar KNN y rankings con muchos scores iguales.

## Estado respecto del enunciado

Están implementados y ejecutados Random, MostPopular, UserKNN, ItemKNN,
FunkSVD, ALS implícito, BPR, Content-Based y dos baselines adicionales
(`BestRated`, `BaselineOnly`).
`BaselineOnly` estima sesgos mediante ALS interno de Surprise, pero **no es** la
factorización ALS para feedback implícito pedida en la tarea; esa corresponde al
método `ALS` separado. Se probaron 20, 50 y 100 factores para ALS/BPR y dos
regularizaciones para ALS. Content-Based compara TF-IDF de ingredientes con los
embeddings BERT provistos, además de perfiles binarios y ponderados por rating.

La taxonomía de estilos culinarios sigue fuera del alcance actual: el informe
habla de diversidad de tags como proxy y no la presenta como equivalencia.

DeepFM es opcional y no se mezcla con la tabla principal mientras no tenga una
corrida bajo este mismo protocolo.

## Artefactos

Las tablas compactas, manifests, hashes y protocolos se guardan en
`informe_assets/data/`. Los archivos detallados por usuario, Top-10 y
predicciones permiten auditar casos concretos, pero son regenerables y se
mantienen fuera de Git para evitar inflar la entrega. La comparación pareada
entre ALS y MostPopular queda en `als_vs_mostpopular_bootstrap.json`.

Las figuras se exportan simultáneamente a PNG, para Markdown, y PDF vectorial,
para el informe final. Los nombres principales son:

- `eda_ratings_time`, `eda_long_tail`, `eda_metadata`;
- `itemknn_sensitivity`, `itemknn_neighbors`,
  `itemknn_ranking_sensitivity`;
- `funksvd_sensitivity`, `funksvd_ranking_sensitivity`;
- `implicit_factor_sensitivity`, `content_based_sensitivity`;
- `prediction_quality`, `ranking_quality`, `ranking_discovery` y
  `ranking_diversity`.
