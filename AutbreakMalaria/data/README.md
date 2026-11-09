# Data

This folder is for the `BdiDATASETS.xlsx` workbook used by the pipeline.

## Expected file

`BdiDATASETS.xlsx` — district-month malaria cases (2022–2024) plus
meteorological and hydrological covariates. Required sheets:

| Sheet | Contents |
|---|---|
| `2022` | District-month cases, header row 2, districts start with `DS ` |
| `2023` | Same as `2022` |
| `2024` | Two-table layout — district months start at column T (19), row 4 |
| `Temperaturemoyen` | Cols E / F / G → Year / Month / Temp |
| `Humiditerelatives  stations` | Cols E / F / G → Year / Month / Humidity |
| `precipitationmensuel30years` | Wide — Year in col E, 12 month cols |
| `Niveau du Lac Tanganyika` | Cols E / F / G → Year / Month / Lake_Level |
| `Niveau d'eau de la RiviereRusiz` | Cols E / F / G → Year / Month / River_Level |

## Placement

Put the workbook at one of:

- `./BdiDATASETS.xlsx`
- `./data/BdiDATASETS.xlsx`   (recommended)
- `~/BdiDATASETS.xlsx`
- `~/Downloads/BdiDATASETS.xlsx`

Or pass the path explicitly:

```python
from srcCode.pipeline import run_pipeline
run_pipeline(data_file="path/to/BdiDATASETS.xlsx")
```

> The workbook is **not** committed to git — see `.gitignore`.
