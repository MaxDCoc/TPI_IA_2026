# Datos

## UCI Heart Disease (dataset principal, a confirmar con la cátedra)

- **Fuente:** UCI Machine Learning Repository, *Heart Disease* (id 45): <https://archive.ics.uci.edu/dataset/45/heart+disease>
- **Cita** (formato que da UCI): Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
- **Artículo de referencia:** Detrano, R., Jánosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.
- **Licencia:** CC BY 4.0 (se puede redistribuir citando la fuente).
- **Descarga:** 04/11/2026, desde `https://archive.ics.uci.edu/static/public/45/heart+disease.zip` (126 KB).

## Archivos incluidos en `data/uci_heart/`

`processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`, `heart-disease.names` (documentación) y `WARNING`.

No se versionan el zip ni los archivos crudos (`cleveland.data`, etc.). El `WARNING` de los autores indica que el crudo `cleveland.data` quedó dañado y que `processed.cleveland.data` es utilizable para la situación de 14 atributos.

## Formato

Sin encabezado, 14 columnas separadas por coma, `?` = valor faltante.

| Columna | Qué es | Tipo propuesto | Codificación |
|---|---|---|---|
| `age` | edad (años) | numérico | |
| `sex` | sexo | nominal | 1 = varón, 0 = mujer |
| `cp` | tipo de dolor de pecho | nominal | 1 angina típica, 2 atípica, 3 no anginoso, 4 asintomático |
| `trestbps` | presión en reposo (mm Hg) | numérico | |
| `chol` | colesterol sérico (mg/dl) | numérico | |
| `fbs` | glucemia en ayunas > 120 mg/dl | nominal | 1 = sí |
| `restecg` | electrocardiograma en reposo | nominal | 0 normal, 1 anomalía ST-T, 2 hipertrofia ventricular |
| `thalach` | frecuencia cardíaca máxima | numérico | |
| `exang` | angina inducida por ejercicio | nominal | 1 = sí |
| `oldpeak` | depresión del ST inducida por ejercicio | numérico | |
| `slope` | pendiente del segmento ST | nominal | 1 ascendente, 2 plana, 3 descendente |
| `ca` | vasos principales coloreados (0–3) | numérico (conteo) | |
| `thal` | resultado del estudio de talio | nominal | 3 normal, 6 defecto fijo, 7 defecto reversible |
| `num` | **objetivo**: diagnóstico | clase | 0 = ausencia; 1–4 = presencia de enfermedad |

Los códigos de `cp`, `restecg`, `slope` y `thal` **no tienen orden**: se declaran nominales para que el árbol ramifique por valor y no corte con umbrales sobre un código.

## Auditoría de los archivos (04/11/2026)

| Archivo | Filas | Clases (`num` 0 / 1 / 2 / 3 / 4) | Faltantes destacados |
|---|---|---|---|
| Cleveland | 303 | 164 / 55 / 36 / 35 / 13 | `ca`: 4, `thal`: 2. Sin duplicados. |
| Hungría | 294 | 188 / 106 / – / – / – | `ca`: 291, `thal`: 266, `slope`: 190 |
| Suiza | 123 | 8 / 48 / 32 / 30 / 5 | `chol` vale 0 en las 123 filas (faltante disfrazado), `ca`: 118 |
| VA Long Beach | 200 | 51 / 56 / 41 / 42 / 10 | `chol` vale 0 en 49 filas, `trestbps`: 56, `ca`: 198 |

Total: 920 filas. Cleveland es el único archivo prácticamente completo.

## Decisiones sobre estos datos

La fuente, la definición del objetivo (binario o varias clases), el tratamiento de faltantes y los tipos de atributo son decisiones del equipo y se van a registrar con su evidencia en `DECISIONES.md`.

## Otra copia: Kaggle (no se usa como principal)

La copia de Kaggle «Heart Disease Dataset» (usuario `johnsmith88`) tiene según su página 1025 filas, objetivo binario y licencia «Unknown». Las discusiones de esa página reportan 723 filas duplicadas y, según un hilo, que son solo datos de Cleveland con el objetivo invertido. Esto último no está verificado contra el archivo.
