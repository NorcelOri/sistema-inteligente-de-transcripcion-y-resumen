"""Pruebas del tokenizador de ROUGE (src/evaluacion.py). Ejecutar: python -m unittest"""

import unittest

from src.evaluacion import _Tokenizador, rouge


class TestRouge(unittest.TestCase):
    def test_tokenizador_conserva_tildes_y_enie(self):
        self.assertEqual(_Tokenizador().tokenize("La reunión del año, ¿sí?"),
                         ["la", "reunión", "del", "año", "sí"])

    def test_tokenizador_sin_tildes_conserva_enie(self):
        self.assertEqual(_Tokenizador(True).tokenize("La reunión del año"),
                         ["la", "reunion", "del", "año"])

    def test_identicos(self):
        r = rouge("la reunión del año", "La reunión del año.")
        self.assertEqual(r["con_tildes"]["rouge1"]["f"], 1.0)
        self.assertEqual(r["con_tildes"]["rougeL"]["f"], 1.0)

    def test_sin_tildes_perdona_acentos(self):
        r = rouge("la reunión del año", "la reunion del año")
        self.assertEqual(r["con_tildes"]["rouge1"]["f"], 0.75)
        self.assertEqual(r["sin_tildes"]["rouge1"]["f"], 1.0)
