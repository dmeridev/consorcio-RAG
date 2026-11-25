"""Números de artículo escritos con letra («treinta y tres a)») a cifra («33 a)»).

La Ley de Contrato de Seguro numera sus artículos con palabras. Para citar y para
comparar en la evaluación queremos siempre el mismo formato: «art. 33 a)».
"""

from __future__ import annotations

import re

_UNIDADES = {
    "uno": 1,
    "un": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}
_ORDINALES = {
    "primero": 1,
    "segundo": 2,
    "tercero": 3,
    "cuarto": 4,
    "quinto": 5,
    "sexto": 6,
    "séptimo": 7,
    "octavo": 8,
    "noveno": 9,
}
_SIMPLES = {
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintidós": 22,
    "veintitrés": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiséis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
_DECENAS = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
# Lo que puede ir detrás del número: «bis», «ter», «quáter» o una letra «a)».
_SUFIJO = re.compile(r"\s+(bis|ter|quáter|[a-z]\))$")


def _palabras_a_numero(texto: str) -> int | None:
    palabras = texto.split()
    total = 0
    if palabras and palabras[0] in ("cien", "ciento"):
        total, palabras = 100, palabras[1:]
    resto = " ".join(palabras)
    if not resto:
        return total or None
    for tabla in (_ORDINALES, _UNIDADES, _SIMPLES):
        if resto in tabla:
            return total + tabla[resto]
    partes = resto.split(" y ")
    if partes[0] in _DECENAS:
        decena = _DECENAS[partes[0]]
        if len(partes) == 1:
            return total + decena
        if len(partes) == 2 and partes[1] in _UNIDADES:
            return total + decena + _UNIDADES[partes[1]]
    return None


def normalizar_numero(numero: str) -> str:
    """'treinta y tres a)' → '33 a)'; '8' → '8'; 'único' → 'único'.

    Si no lo reconoce, lo devuelve tal cual: mejor una cita rara que un número inventado.
    """
    numero = " ".join(numero.split()).lower()
    sufijo = ""
    if m := _SUFIJO.search(numero):
        sufijo = " " + m.group(1)
        numero = numero[: m.start()]
    if numero.isdigit() or numero == "único":
        return numero + sufijo
    valor = _palabras_a_numero(numero)
    return (str(valor) if valor is not None else numero) + sufijo
