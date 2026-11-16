"""
Enfermedad cardíaca (UCI Heart Disease, Cleveland).

GUÍA DE LECTURA
---------------
De los archivos de data/uci_heart/ solo se usa processed.cleveland.data: según el aviso de los autores (WARNING),
el archivo crudo cleveland.data quedó dañado y los de Hungría, Suiza y Long Beach tienen muchísimos faltantes
(ver la auditoría en DATOS.md). El archivo no trae encabezado: 14 columnas separadas por coma y «?» para los faltantes.

Para que el árbol se pueda leer sin un diccionario al lado:
  - las columnas se renombran en español (el nombre original de UCI queda en la descripción de cada atributo);
  - los códigos de los atributos nominales se traducen (por ejemplo, tipo_dolor = 4 pasa a «sin síntomas»).
El significado de cada código sale de heart-disease.names, la documentación oficial del dataset.
"""
from __future__ import annotations

from pathlib import Path

from .loader import Dataset, load_dataset
from .schema import DatasetSpec, Scenario

CITATION = ("Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X")

# (nombre en español, nombre original en UCI, descripción para la interfaz)
ATTRIBUTES = [
    ("edad", "age", "Edad del paciente, en años."),
    ("sexo", "sex", "Sexo del paciente."),
    ("tipo_dolor", "cp", "Tipo de dolor de pecho que presenta el paciente."),
    ("presion_reposo", "trestbps", "Presión arterial en reposo, en mm Hg, medida al ingresar al hospital."),
    ("colesterol", "chol", "Colesterol en sangre, en mg/dl."),
    ("glucemia_ayunas", "fbs", "Indica si el azúcar en sangre en ayunas supera los 120 mg/dl."),
    ("ecg_reposo", "restecg", "Resultado del electrocardiograma en reposo."),
    ("frecuencia_max", "thalach", "Frecuencia cardíaca máxima alcanzada durante la prueba de esfuerzo, en latidos por minuto."),
    ("angina_ejercicio", "exang", "Indica si el ejercicio le provoca angina (dolor de pecho) al paciente."),
    ("depresion_st", "oldpeak", "Cuánto baja el segmento ST del electrocardiograma con el ejercicio, comparado con el reposo."),
    ("pendiente_st", "slope", "Forma que toma el segmento ST en el momento de máximo esfuerzo."),
    ("vasos_coloreados", "ca", "Cantidad de vasos sanguíneos principales (de 0 a 3) que se ven en la fluoroscopía."),
    ("talio", "thal", "Resultado de la prueba de esfuerzo con talio, que muestra cómo llega la sangre al corazón."),
]
TARGET = "diagnostico"
COLUMNS = [name for name, _, _ in ATTRIBUTES] + [TARGET]
DESCRIPTIONS = {name: f"{text} En UCI se llama «{original}»." for name, original, text in ATTRIBUTES}

# Qué significa cada código (según heart-disease.names). vasos_coloreados es un conteo, así que queda numérico.
VALUE_LABELS = {
    "sexo": {0: "mujer", 1: "hombre"},
    "tipo_dolor": {1: "angina típica", 2: "angina atípica", 3: "dolor no anginoso", 4: "sin síntomas"},
    "glucemia_ayunas": {0: "120 mg/dl o menos", 1: "más de 120 mg/dl"},
    "ecg_reposo": {0: "normal", 1: "anomalía de la onda ST-T", 2: "hipertrofia ventricular"},
    "angina_ejercicio": {0: "no", 1: "sí"},
    "pendiente_st": {1: "ascendente", 2: "plana", 3: "descendente"},
    "talio": {3: "normal", 6: "defecto fijo", 7: "defecto reversible"},
}

SCENARIOS = {
    "binario": Scenario(
        "binario", mapping={0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
        class_names={0: "sin enfermedad", 1: "con enfermedad"},
        goal="predecir si el paciente tiene o no una enfermedad cardíaca",
        description="La columna original va de 0 a 4: el 0 indica que no hay enfermedad y los valores de 1 a 4 indican "
                    "que sí la hay. Según la documentación del dataset, los experimentos con estos datos se concentraron "
                    "en distinguir la presencia de la enfermedad (valores 1 a 4) de su ausencia (valor 0), así que se "
                    "agrupan los valores de 1 a 4 en una sola clase. Además, así las dos clases quedan parejas (164 y "
                    "139 casos), mientras que con las cinco clases originales la más chica tendría solo 13 casos."),
    "multiclase": Scenario(
        "multiclase", mapping=None,
        class_names={0: "sin enfermedad", 1: "valor 1", 2: "valor 2", 3: "valor 3", 4: "valor 4"},
        goal="predecir el valor original del diagnóstico, de 0 a 4",
        description="Se predice la columna original, con sus cinco valores: el 0 indica que no hay enfermedad y los "
                    "valores de 1 a 4, que sí la hay. La documentación del dataset no explica qué distingue a cada uno "
                    "de esos cuatro valores."),
}

HEART_SPEC = DatasetSpec(
    name="Enfermedad cardíaca", target=TARGET, columns=COLUMNS, header=False,
    nominal=list(VALUE_LABELS), value_labels=VALUE_LABELS, descriptions=DESCRIPTIONS, scenarios=SCENARIOS,
    source="el archivo processed.cleveland.data del UCI Machine Learning Repository (Heart Disease), que reúne los "
           "pacientes de la Cleveland Clinic Foundation. Es el único archivo de la colección que los autores "
           "señalan como confiable y que está prácticamente completo",
    citation=CITATION,
    purpose="Es un problema real de diagnóstico médico que mezcla atributos numéricos (como la edad o el colesterol) "
            "con atributos nominales (como el tipo de dolor de pecho), e incluye algunos valores faltantes. Por eso "
            "permite probar todas las capacidades del árbol C4.5 en un caso realista.")


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "uci_heart"


def load_heart_dataset(data_dir=None, scenario: str = "binario", **kwargs) -> Dataset:
    """Carga processed.cleveland.data como Dataset. Por defecto, con el objetivo binario (con o sin enfermedad)."""
    path = (Path(data_dir) if data_dir else default_data_dir()) / "processed.cleveland.data"
    return load_dataset(path, HEART_SPEC, scenario=scenario, **kwargs)
