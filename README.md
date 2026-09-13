# Tarea 1 — IIC3633 Sistemas Recomendadores

**Equipo:** Grupo 18 (eight nine)

**Integrantes:** Jorge Cartagena y Santiago Valenzuela

Este repositorio contiene el desarrollo reproducible de la Tarea 1 sobre
recomendación de recetas. La documentación técnica, el orden de ejecución y el
protocolo de evaluación se describen en [`tarea_01/README.md`](tarea_01/README.md).

## Estructura principal

- `tarea_01/notebooks/`: notebooks ejecutados de EDA y recomendación.
- `tarea_01/*.py`: implementación compartida de experimentos y evaluación.
- `tarea_01/informe_assets/`: figuras y evidencia compacta del informe.
- `tarea_01/INFORME_BORRADOR.md`: fuente editable del informe que debe exportarse
  a PDF antes de la entrega.
- `tarea_01/USO_IA.md`: registro de las conversaciones de IA que deben adjuntarse.
- `tarea_01/bonus_submission/`: archivos validados para el bonus de competencia.
- `Tarea_1_RecSys_2026_2.pdf`: enunciado de la tarea.

Los directorios locales `tarea_01/dataset/`, `tarea_01/results/` y
`tarea_01/figures/` no forman parte de la entrega. Tampoco deben incluirse
`.venv/`, cachés ni archivos temporales.

## Entorno

Se requiere Python 3.12 y [`uv`](https://docs.astral.sh/uv/). Desde la raíz del
repositorio:

```powershell
pip install uv
uv sync --dev
```

El archivo `uv.lock` fija las versiones resueltas. No es necesario crear el
entorno manualmente: `uv sync --dev` crea o actualiza `.venv` de forma
reproducible.

## Datos

Los datos proporcionados por el curso deben ubicarse en `tarea_01/dataset/`.
Como mínimo, la ejecución principal espera:

- `train.csv`, `validation.csv` e `item_metadata.csv`;
- `food_bert_embeddings*.npy` y `food_item_id_to_idx*.npy`;
- las plantillas de competencia, solo si se regenera el bonus.

El dataset no se versiona ni se incluye en el paquete de entrega.

## Reproducción y entrega

Los comandos, el orden obligatorio de notebooks, las métricas y el contenido
recomendado del paquete están detallados en
[`tarea_01/README.md`](tarea_01/README.md). Antes de subir los archivos se debe:

1. exportar el informe final a PDF;
2. conservar los outputs de los notebooks;
3. verificar el acceso a los enlaces de [`tarea_01/USO_IA.md`](tarea_01/USO_IA.md);
4. excluir datos, resultados intermedios, entornos y cachés.
