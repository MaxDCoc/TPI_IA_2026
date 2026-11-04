"""
Datasets de juguete con resultado conocido. Sirven para TESTS y DEMOS: los números se pueden
verificar a mano contra el libro. No son datos del trabajo práctico.
"""
from __future__ import annotations

import pandas as pd


def play_tennis():
    """Play Tennis (Quinlan 1986; Mitchell 1997, cap. 3): 14 casos, 4 atributos nominales, 2 clases.

    Ganancias esperadas en la raíz (en bits): Outlook 0.246, Humidity 0.151, Wind 0.048,
    Temperature 0.029. Árbol esperado: Outlook en la raíz; Overcast -> Yes; Sunny -> Humidity;
    Rain -> Wind.
    """
    rows = [
        ("Sunny", "Hot", "High", "Weak", "No"),
        ("Sunny", "Hot", "High", "Strong", "No"),
        ("Overcast", "Hot", "High", "Weak", "Yes"),
        ("Rain", "Mild", "High", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"),
        ("Sunny", "Mild", "High", "Weak", "No"),
        ("Sunny", "Cool", "Normal", "Weak", "Yes"),
        ("Rain", "Mild", "Normal", "Weak", "Yes"),
        ("Sunny", "Mild", "Normal", "Strong", "Yes"),
        ("Overcast", "Mild", "High", "Strong", "Yes"),
        ("Overcast", "Hot", "Normal", "Weak", "Yes"),
        ("Rain", "Mild", "High", "Strong", "No"),
    ]
    df = pd.DataFrame(rows, columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"])
    return df.drop(columns="PlayTennis"), df["PlayTennis"]


def liu_loans():
    """Préstamos (Liu, Web Data Mining, tabla 3.1): 15 casos, 4 atributos nominales, 2 clases.

    Ganancias esperadas en la raíz (en bits): Age 0.083, Has_job 0.324, Own_house 0.420,
    Credit_rating 0.363. Árbol esperado: Own_house en la raíz; true -> Yes; false -> Has_job
    (true -> Yes, false -> No).
    """
    rows = [
        ("young", "false", "false", "fair", "No"),
        ("young", "false", "false", "good", "No"),
        ("young", "true", "false", "good", "Yes"),
        ("young", "true", "true", "fair", "Yes"),
        ("young", "false", "false", "fair", "No"),
        ("middle", "false", "false", "fair", "No"),
        ("middle", "false", "false", "good", "No"),
        ("middle", "true", "true", "good", "Yes"),
        ("middle", "false", "true", "excellent", "Yes"),
        ("middle", "false", "true", "excellent", "Yes"),
        ("old", "false", "true", "excellent", "Yes"),
        ("old", "false", "true", "good", "Yes"),
        ("old", "true", "false", "good", "Yes"),
        ("old", "true", "false", "excellent", "Yes"),
        ("old", "false", "false", "fair", "No"),
    ]
    df = pd.DataFrame(rows, columns=["Age", "Has_job", "Own_house", "Credit_rating", "Class"])
    return df.drop(columns="Class"), df["Class"]
