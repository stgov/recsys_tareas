# Archivos para el bonus de competencia

**Equipo:** Grupo 18 (eight nine)

**Integrantes:** Jorge Cartagena y Santiago Valenzuela

Estos archivos son independientes del paquete principal de la tarea. Para
participar en la competencia, subir en <https://contest-recsys.vercel.app/>:

- `competicion_rating_completado.csv` en **Predicción de rating**.
- `competicion_ranking.json` en **Predicción de ranking**.

`submission_manifest.json` conserva el protocolo, las métricas de desarrollo y
los hashes SHA-256 de los archivos. Sirve para comprobar que no cambiaron, pero
no se sube a la plataforma.

Los archivos se regeneran desde la raíz de `recsys_tareas` con:

```powershell
uv run python -m tarea_01.competition_submission
```

La regeneración vuelve a entrenar los modelos usando los datos locales. Si solo
se desea entregar los archivos ya validados, no es necesario ejecutar este
comando.
