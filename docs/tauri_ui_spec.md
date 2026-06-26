# TraTrac — Especificación de UI del cliente Tauri

Editor de configuración para generar un `run.toml` válido y lanzar `tratrac`.

La fuente de verdad del esquema es `src/tratrac/application/config.py` (`RunConfig.resolve`)
y la plantilla `tratrac.example.toml`. Esta UI es un **editor fiel** de ese esquema:
el paquete **no tiene defaults ocultos** — toda clave es obligatoria — así que la UI
debe (a) preconfigurar valores sensatos y (b) reflejar exactamente las validaciones del
resolver.

---

## Principios de diseño (leer antes de implementar)

1. **La UI es la proveedora de defaults que el paquete deliberadamente no tiene.**
   Precargar cada campo con los valores de `tratrac.example.toml`. El operador edita
   3 campos, no 20. El archivo escrito sigue siendo completo y explícito, así que la
   garantía de reproducibilidad se mantiene.

2. **No reimplementar la validación en JS/Rust — derivará de `resolve()`.**
   Las restricciones inline son solo pistas de UX. La autoridad debe ser un comando
   `tratrac --check` (a añadir) que corra `RunConfig.resolve` y emita los problemas
   agregados de `ConfigError` como JSON; Tauri lo invoca por shell. Así
   `application/config.py` sigue siendo la única fuente de verdad.

3. **`--force` es estado de la acción de ejecución, NO un campo del formulario.**
   La política de sobrescritura nunca afecta a las trayectorias, por eso no es clave de
   config. Modelar como checkbox en el botón "Ejecutar" y pasarlo como flag CLI; nunca
   serializarlo al TOML.

4. **Habilitación condicional = guardas de coherencia del resolver.**
   Deshabilitar/colapsar los controles dependientes en vez de permitir escribir una
   contradicción que el run rechazaría.

---

## Campos por sección

### `[input]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `video` | selector de archivo (mp4…) | debe existir en disco (la CLI lo verifica) |
| `process_fps` | número + toggle "cada frame" | `>= 0`; `0.0` = procesar cada frame |

### `[detector]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `name` | dropdown | enum: `yolov8_visdrone` \| `rt_detr` |
| `checkpoint` | texto (precargar según `name`) | id de repo HF |
| `conf` | slider | `[0, 1]` |
| `filename` | texto | obligatorio incluso para `rt_detr` (que lo ignora) |

### `[runtime]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `device` | segmented + spinner de índice | `cpu` \| `mps` \| `cuda[:N]` |

### `[calibration]` — **radio de "exactamente uno"**
Control con forma de árbol; un radio (no campos independientes) es la affordance correcta
porque especificar A **y** B es error.

- **Opción A** — `meters_per_pixel`: número `> 0`.
- **Opción B** — `drone_model`: dropdown **poblado desde `known_models()`** (no hardcodear)
  + sub-elección de altitud:
  - `altitude_m`: número `> 0`, **o**
  - `srt`: selector de archivo (sidecar DJI `.SRT`).

### `[ego_motion]`
`enabled` (toggle) gatea los 5 parámetros; colapsarlos cuando está off.

| Clave | Widget | Restricción |
| --- | --- | --- |
| `enabled` | toggle | — |
| `n_features` | int | `> 0` |
| `match_ratio` | slider | `(0, 1)` |
| `min_matches` | int | `>= 2` |
| `ransac_threshold` | número | `> 0` |
| `min_anchor_overlap` | slider | `(0, 1)` |

### `[tracker]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `det_thresh` | slider | `[0, 1]` (típicamente por debajo de `detector.conf`) |

### `[export]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `out` | selector de guardado (`.parquet`) | no vacío; salida primaria del run |
| `transform_csv` | path opcional, "" = off | **requiere `ego_motion.enabled`** |
| `anchors_dir` | directorio opcional, "" = off | **requiere `ego_motion.enabled`** |

### `[window]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `start` | input de timecode | `SS` / `MM:SS` / `HH:MM:SS`; "" = inicio del clip |
| `end` | input de timecode | "" = fin del clip; `end > start` |

### `[run]`
| Clave | Widget | Restricción |
| --- | --- | --- |
| `timing_csv` | path opcional, "" = off | CSV de timings por frame |

---

## Acción de ejecución (fuera del TOML)

- **Botón "Ejecutar"** → escribe el TOML y lanza `tratrac --config <archivo>`.
- **Checkbox "Sobrescribir salidas"** → añade `--force` al comando (no va al TOML).
- Mostrar stdout/stderr del proceso (la CLI ya reporta progreso a stderr).

## Pos-proceso (siguiente pantalla, opcional)

El run es solo percepción: produce el record Parquet, no un `.trj`. Un `.trj` se obtiene
con `tratrac-postprocess RECORD --out run.trj [...]`. Si la UI lo cubre, sus inputs serían
otra pantalla (filtros de exclusión, `--calibration`, parámetros de suavizado) — fuera del
alcance de este editor de config.
