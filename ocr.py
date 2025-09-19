# ocr.py
"""
OCR con Tesseract (Python)
Lee una imagen y devuelve todo el texto detectado.
Soporta preprocesado básico para mejorar resultados.
"""

import sys
import os
from PIL import Image
import numpy as np
import cv2
import pytesseract

# Si en Windows no está en PATH, descomenta y ajusta:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(path, show=False):
    """
    Preprocesado simple con OpenCV para mejorar OCR:
    - lectura en escala de grises
    - filtro bilateral para reducir ruido
    - umbral adaptativo para mejorar contraste
    - (opcional) reescalar la imagen
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    # reescalar (opcional) si la imagen es muy pequeña
    height, width = img.shape[:2]
    scale = 1.0
    if max(height, width) < 1000:
        scale = 2.0
        img = cv2.resize(img, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reducir ruido manteniendo bordes
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # umbral adaptativo
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)

    if show:
        cv2.imshow("preprocessed", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh

def image_to_text(path, lang='spa', do_preprocess=True, custom_config=''):
    if do_preprocess:
        img = preprocess_image(path, show=False)  # no abrir ventana
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.open(path)
        if pil_img.mode not in ("L", "RGB"):
            pil_img = pil_img.convert("RGB")

    default_config = r'--oem 1 --psm 6'  # psm 6 suele ir bien para bloques
    config = (default_config + ' ' + custom_config).strip()

    return pytesseract.image_to_string(pil_img, lang=lang, config=config)

def main():
    if len(sys.argv) < 2:
        print("Uso: python ocr.py ruta/a/imagen.jpg [--no-preprocess] [--lang=spa]")
        sys.exit(1)

    image_path = sys.argv[1]
    do_pre = True
    lang = 'spa'
    extra = ''

    for arg in sys.argv[2:]:
        if arg == '--no-preprocess':
            do_pre = False
        if arg.startswith('--lang='):
            lang = arg.split('=', 1)[1]
        if arg.startswith('--config='):
            extra = arg.split('=', 1)[1]

    try:
        text = image_to_text(image_path, lang=lang, do_preprocess=do_pre, custom_config=extra)
        print("---- TEXTO DETECTADO ----\n")
        print(text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
