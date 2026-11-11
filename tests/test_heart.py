"""Enfermedad cardíaca (UCI Heart Disease, Cleveland): carga, traducción de códigos y uso desde la interfaz."""
import pytest

from tp_ia.c45 import C45DecisionTree
from tp_ia.data import load_heart_dataset
from tp_ia.web import LabService


@pytest.fixture(scope="module")
def heart():
    return load_heart_dataset()


def test_cleveland_has_the_documented_shape_and_missing_values(heart):
    assert heart.X.shape == (303, 13)
    missing = heart.X.isna().sum()
    assert missing[missing > 0].to_dict() == {"vasos_coloreados": 4, "talio": 2}   # como dice la auditoría de DATOS.md


def test_the_binary_target_groups_values_1_to_4_as_disease(heart):
    assert heart.report["class_counts"] == {0: 164, 1: 139}
    assert heart.class_names == {0: "sin enfermedad", 1: "con enfermedad"}
    d02 = [d for d in heart.log if d.id == "D-02"][-1]
    assert d02.chosen == "predecir si el paciente tiene o no una enfermedad cardíaca"


def test_the_multiclass_scenario_keeps_the_original_five_values():
    multi = load_heart_dataset(scenario="multiclase")
    assert multi.report["class_counts"] == {0: 164, 1: 55, 2: 36, 3: 35, 4: 13}


def test_nominal_codes_are_replaced_by_their_meaning(heart):
    assert set(heart.X["tipo_dolor"]) == {"angina típica", "angina atípica", "dolor no anginoso", "sin síntomas"}
    assert set(heart.X["sexo"]) == {"mujer", "hombre"}
    assert set(heart.X["talio"].dropna()) == {"normal", "defecto fijo", "defecto reversible"}
    assert heart.feature_types["vasos_coloreados"] == "numeric"          # es un conteo de 0 a 3, no un código
    assert sum(t == "nominal" for t in heart.feature_types.values()) == 7
    d05 = next(d for d in heart.log if d.id == "D-05")
    assert "talio" in d05.evidence["códigos reemplazados por su significado"]


def test_every_attribute_has_a_plain_language_description(heart):
    assert set(heart.descriptions) == set(heart.X.columns)
    assert "«cp»" in heart.descriptions["tipo_dolor"]                    # se menciona el nombre original de UCI


def test_the_tree_reads_with_words_instead_of_codes(heart):
    text = C45DecisionTree().fit(heart.X, heart.y, feature_types=heart.feature_types).export_text()
    assert "talio = " in text and "4.0" not in text


def test_heart_is_first_in_the_lab_and_shows_descriptions_and_class_split():
    lab = LabService()
    assert lab.datasets()["datasets"][0]["key"] == "heart"
    info = lab.select("heart")
    assert info["n_rows"] == 303 and [c["name"] for c in info["classes"]] == ["sin enfermedad", "con enfermedad"]
    pain = next(f for f in info["features"] if f["name"] == "tipo_dolor")
    assert pain["description"] and sum(pain["by_class"]["sin síntomas"]) == pain["counts"]["sin síntomas"]
    assert lab.train()["tree"]["root"]["leaf"] is False
