"""Pruebas de la limpieza de muletillas (src/texto.py). Ejecutar: python -m unittest"""

import unittest

from src.texto import limpiar_texto, limpiar_texto_con_registro


class TestLimpiezaMuletillas(unittest.TestCase):
    def test_marcador_bueno_inicio_de_oracion(self):
        self.assertEqual(limpiar_texto("Quizás puedes compartir... Bueno, mañana nos compartes lo que es"),
                         "Quizás puedes compartir. Mañana nos compartes lo que es")

    def test_repetida(self):
        self.assertEqual(limpiar_texto("Podríamos utilizar únicamente su su código de descuento."),
                         "Podríamos utilizar únicamente su código de descuento.")

    def test_cola_no(self):
        self.assertEqual(limpiar_texto("Pero si escribimos un poco más, ¿no? Como me dijiste"),
                         "Pero si escribimos un poco más. Como me dijiste")

    def test_y_bueno(self):
        self.assertEqual(limpiar_texto("Estamos coordinando con Untels. Y bueno, a ver más o menos qué otras"),
                         "Estamos coordinando con Untels. Y a ver más o menos qué otras")

    def test_digamos_cierra_oracion(self):
        self.assertEqual(limpiar_texto("Mayor cantidad de participantes, digamos. Orientarlos o tratar de llamar"),
                         "Mayor cantidad de participantes. Orientarlos o tratar de llamar")

    def test_no_toca_palabras_con_sentido(self):
        texto = "Me parece bueno este evento, pues vamos."
        self.assertEqual(limpiar_texto(texto), texto)

    def test_registro_de_borrados(self):
        _, borrados = limpiar_texto_con_registro("Podemos usar su su código, ¿no?")
        self.assertEqual({b["tipo"] for b in borrados}, {"repetida", "cola ¿no?"})


if __name__ == "__main__":
    unittest.main()
