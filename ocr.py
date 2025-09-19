#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR con Tesseract para imágenes y PDFs, con extracción de CURP y nombre.

Flujo:
1) Dado un archivo (imagen o PDF), obtener el texto:
   - Si es imagen -> OCR directo (con preprocesado opcional).
   - Si es PDF -> intentar texto "nativo" (pdfplumber). Si no hay, rasterizar páginas y hacer OCR.
2) Del texto bruto, extraer:
   - CURP (regex laxa de 18 caracteres).
   - Nombre (por etiquetas típicas o heurística cercana a la CURP).
3) Mostrar por consola el texto detectado y/o un JSON con {"curp","nombre"}.
"""

import sys
import os
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytesseract
import pdfplumber
from pdf2image import convert_from_path


# ---------------------------------------------------------------------------
# 1) PREPROCESADO DE IMÁGENES
# ---------------------------------------------------------------------------
def preprocess_image(path_or_ndarray, show: bool = False) -> np.ndarray:
    """
    Preprocesa una imagen para mejorar la lectura por OCR:
      - Acepta una ruta (str/Path) o un array de OpenCV (BGR/GRAY).
      - Reescala si es pequeña (< 1000px lado mayor).
      - Convierte a escala de grises, filtra ruido y aplica umbral adaptativo.

    Returns:
        ndarray (uint8): imagen binarizada/umbralizada lista para OCR.

    Args:
        path_or_ndarray: str/Path a la imagen o ndarray BGR/GRAY.
        show (bool): si True, abre una ventana con el resultado (debug).
    """
    # Cargar imagen desde ruta o usar el ndarray directamente
    if isinstance(path_or_ndarray, (str, os.PathLike)):
        img = cv2.imread(str(path_or_ndarray))
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path_or_ndarray}")
    else:
        img = path_or_ndarray
        if img is None:
            raise ValueError("ndarray vacío en preprocess_image")

    # Reescalar si es muy pequeña (mejora OCR)
    h, w = img.shape[:2]
    if max(h, w) < 1000:
        img = cv2.resize(img, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_LINEAR)

    # Escala de grises (si venía en color)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Filtro bilateral: reduce ruido preservando bordes (mejor para texto)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Umbral adaptativo gaussiano: incrementa contraste texto/fondo
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    if show:
        cv2.imshow("preprocessed", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh


# ---------------------------------------------------------------------------
# 2) OCR PARA IMÁGENES
# ---------------------------------------------------------------------------
def image_to_text(
    path: str,
    lang: str = "spa",
    do_preprocess: bool = True,
    custom_config: str = "",
) -> str:
    """
    Aplica OCR a una imagen.

    - Si `do_preprocess=True`, ejecuta preprocess_image() y pasa a PIL.
    - Config por defecto: '--oem 1 --psm 6' (modo LSTM y layout de bloque).

    Args:
        path: ruta a la imagen.
        lang: lenguaje del OCR (instala los datos de Tesseract si hace falta).
        do_preprocess: si True, preprocesa la imagen.
        custom_config: flags adicionales de Tesseract (p.ej. '--psm 7').

    Returns:
        str: texto reconocido por Tesseract.
    """
    if do_preprocess:
        img = preprocess_image(path, show=False)  # ndarray
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.open(path)
        # Asegurar formato compatible (algunos modos de PIL)
        if pil_img.mode not in ("L", "RGB"):
            pil_img = pil_img.convert("RGB")

    default_config = r"--oem 1 --psm 6"
    config = (default_config + " " + custom_config).strip()
    return pytesseract.image_to_string(pil_img, lang=lang, config=config)


# ---------------------------------------------------------------------------
# 3) TEXTO DESDE PDF (NATIVO -> OCR SI HACE FALTA)
# ---------------------------------------------------------------------------
def pdf_to_text(
    pdf_path: str,
    lang: str = "spa",
    do_preprocess: bool = True,
    custom_config: str = "",
) -> str:
    """
    Extrae texto desde PDF:
      1) Intenta texto "nativo" (pdfplumber). Ideal para PDFs generados.
      2) Si no hay texto útil, rasteriza páginas y aplica OCR con Tesseract.

    Args:
        pdf_path: ruta al PDF.
        lang, do_preprocess, custom_config: mismas ideas que image_to_text.

    Returns:
        str: texto del PDF (nativo u obtenido via OCR).
    """
    # 1) Texto nativo (más rápido y exacto si existe)
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_chunks.append(page_text)
    except Exception:
        # Si falla pdfplumber, pasamos directo a OCR
        pass

    native_text = "\n".join(text_chunks).strip()
    if len(native_text) >= 20:  # Umbral chico para decidir “ya hay texto”
        return native_text

    # 2) OCR por página (requiere poppler instalado para convert_from_path)
    ocr_texts = []
    try:
        # DPI 300 suele ser suficiente; puedes subir a 400 si el texto está pequeño
        pages = convert_from_path(pdf_path, dpi=300)
        for pil_img in pages:
            # PIL (RGB) -> ndarray (BGR) para reusar el preprocesado
            arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            pre = preprocess_image(arr, show=False)
            pil_pre = Image.fromarray(pre)

            cfg = (r"--oem 1 --psm 6 " + custom_config).strip()
            page_text = pytesseract.image_to_string(pil_pre, lang=lang, config=cfg)
            ocr_texts.append(page_text)
    except Exception as e:
        raise RuntimeError(f"OCR de PDF falló: {e}")

    return "\n".join(ocr_texts)


# ---------------------------------------------------------------------------
# 4) DESPACHADOR SEGÚN EXTENSIÓN (PDF o IMAGEN)
# ---------------------------------------------------------------------------
def file_to_text(
    path: str, lang: str = "spa", do_preprocess: bool = True, custom_config: str = ""
) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return pdf_to_text(path, lang=lang, do_preprocess=do_preprocess, custom_config=custom_config)
    return image_to_text(path, lang=lang, do_preprocess=do_preprocess, custom_config=custom_config)


# ---------------------------------------------------------------------------
# 5) EXTRACCIÓN DE CURP Y NOMBRE
# ---------------------------------------------------------------------------

# Regex laxa de CURP: 18 caracteres con estructura general.
# Nota: no valida a detalle códigos de estado ni verificador; útil para “encontrar”.
CURP_REGEX_LOOSE = re.compile(r"\b[A-ZÑ]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}\b")

def extract_curp_and_name(raw_text: str) -> dict[str, str | None]:
    """
    A partir de texto OCR (o nativo), intenta extraer:
      - CURP: primera coincidencia que cumpla el patrón
      - Nombre: por etiquetas conocidas ('NOMBRE', 'NOMBRE(S)', etc.) o,
                si no se encuentra etiqueta, línea en MAYÚSCULAS cercana a la CURP.

    Heurística de nombre:
      - Toma la parte a la derecha/abajo de la etiqueta en la misma línea.
      - Si no hay etiqueta, busca alrededor de la línea que contiene la CURP
        otra línea en mayúsculas suficientemente larga y la usa como nombre.

    Returns:
        dict con claves:
            {"curp": <str|None>, "nombre": <str|None>}
    """
    # Normalizar retornos de carro y trabajar en mayúsculas para coincidencias robustas
    text = raw_text.replace("\r", "")
    U = text.upper()

    # --- CURP ---
    curp = None
    m = CURP_REGEX_LOOSE.search(U)
    if m:
        curp = m.group(0)

    # --- Nombre (por etiquetas) ---
    nombre = None
    label_patterns = [
        r"(?:(?:NOMBRE\(S\)|NOMBRES|NOMBRE)\s*[:\-]?\s*)(.+)",
        r"(?:NOMBRE\s+DEL\s+TITULAR\s*[:\-]?\s*)(.+)",
        r"(?:NOMBRE\s+COMPLETO\s*[:\-]?\s*)(.+)",
        r"(?:NOMBRE\s*[:\-]?\s*)(.+)",
        r"(?:NOMBRE\(S\)\s*[:\-]?\s*)(.+)",
        r"(?:PRIMER\s+APELLIDO\s*[:\-]?\s*)(.+)",
    ]
    for pat in label_patterns:
        mm = re.search(pat, U)
        if mm:
            # Tomamos el texto que sigue a la etiqueta, hasta fin de línea;
            # limpiamos espacios repetidos y lo “Title-case”.
            line = mm.group(1).strip()
            line = line.splitlines()[0].strip()
            nombre = re.sub(r"\s{2,}", " ", line).title()
            break

    # --- Nombre (heurística cercana a la CURP) ---
    if not nombre and curp:
        lines = U.splitlines()
        # Índice de la línea donde aparece la CURP
        idx = next((i for i, ln in enumerate(lines) if curp in ln), None)
        if idx is not None:
            # Ventana +/- 5 líneas alrededor
            window = lines[max(0, idx - 5) : idx + 6]
            # Candidatas: líneas en MAYÚSCULAS “largas”, sin la CURP
            candidates = [ln.strip() for ln in window if len(ln.strip()) >= 8]
            candidates = [c for c in candidates if re.fullmatch(r"[A-ZÁÉÍÓÚÑ\s\.'\-]+", c)]
            candidates = [c for c in candidates if curp not in c]
            if candidates:
                nombre = candidates[0].title()

    return {"curp": curp, "nombre": nombre}


# ---------------------------------------------------------------------------
# 6) CLI / ENTRADA POR TERMINAL
# ---------------------------------------------------------------------------
def main():
    """
    Uso:
        python ocr.py ruta/a/archivo.(jpg|png|pdf) [--no-preprocess] [--lang=spa]
                      [--config='--psm 6'] [--json]

    Ejemplos:
        python ocr.py samples/ine.jpg --lang=spa --config="--psm 6" --json
        python ocr.py docs/curp.pdf --lang=spa --json
    """
    if len(sys.argv) < 2:
        print("Uso: python ocr.py ruta/a/archivo.(jpg|png|pdf) [--no-preprocess] [--lang=spa] [--config='--psm 6'] [--json]")
        sys.exit(1)

    file_path = sys.argv[1]
    do_pre = True            # preprocesar imágenes por defecto
    lang = "spa"             # idioma OCR por defecto
    extra = ""               # flags extra para tesseract
    as_json = "--json" in sys.argv  # salida compacta en JSON

    # Parseo simple de flags
    for arg in sys.argv[2:]:
        if arg == "--no-preprocess":
            do_pre = False
        elif arg.startswith("--lang="):
            lang = arg.split("=", 1)[1]
        elif arg.startswith("--config="):
            extra = arg.split("=", 1)[1]

    try:
        # 1) Obtener texto (según tipo de archivo)
        text = file_to_text(file_path, lang=lang, do_preprocess=do_pre, custom_config=extra)

        # 2) Extraer CURP y nombre
        result = extract_curp_and_name(text)

        # 3) Mostrar salida
        if as_json:
            import json
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("---- TEXTO DETECTADO ----\n")
            print(text)
            print("\n---- EXTRACCIÓN ----")
            print(f"CURP:   {result['curp'] or 'No encontrada'}")
            print(f"Nombre: {result['nombre'] or 'No encontrado'}")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
