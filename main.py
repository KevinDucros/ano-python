import io
import re
from typing import List, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np

app = FastAPI(title="PDF Anonymizer")

# --- Heuristiques / regex ---

SHORT_WORDS = {"de", "du", "la", "le", "des", "d", "l", "et"}
COMPANY_KEYWORDS = {"entreprise", "societe", "société", "sarl", "sas", "sa", "eurl", "auto-entrepreneur"}
ADDRESS_KEYWORDS = {"rue", "avenue", "av.", "bd", "boulevard", "impasse", "allée", "allee", "route", "chemin", "place"}

PERSON_LABELS = {"nom", "prénom", "prenom", "titulaire", "client"}
ADDRESS_LABELS = {"adresse"}

IBAN_MIN_DIGITS = 20  # un peu en dessous de la réalité pour tolérer OCR


def normalize(text: str) -> str:
    return text.strip().lower().replace(":", "").replace(".", "")


def is_person_word(word: str) -> bool:
    """Heuristique 'nom/prénom' : majuscule, au moins 2 lettres, pas un petit mot courant."""
    w = word.strip()
    if not w:
        return False

    if normalize(w) in SHORT_WORDS:
        return False

    # ex: "Dupont", "Jean-Pierre"
    is_capitalized = re.match(r"^[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][a-zàâäçéèêëîïôöùûüÿ'-]{1,}$", w) is not None
    # ex: "MARTIN"
    is_all_caps = re.match(r"^[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ'-]{2,}$", w) is not None

    return bool(is_capitalized or is_all_caps)


def is_address_like_line(text: str) -> bool:
    """Détecte une ligne qui ressemble à une adresse (du client)."""
    t = text.lower()
    # numéro de rue + mot
    if re.search(r"\b\d{1,3}\s+\S+", t):
        return True
    # code postal français
    if re.search(r"\b\d{5}\b", t):
        return True
    # mots-clés type voie
    if any(k in t for k in ADDRESS_KEYWORDS):
        return True
    return False


def collect_lines(data: Dict) -> List[List[Dict]]:
    """
    Regroupe les mots Tesseract (image_to_data) par ligne.
    Retourne une liste de lignes, chaque ligne = liste de dict {text,left,top,width,height}
    """
    n = len(data["text"])
    lines_map: Dict[Tuple[int, int, int], List[Dict]] = {}

    for i in range(n):
        text = data["text"][i]
        if not text or text.strip() == "":
            continue
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1
        if conf < 0:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        word_info = {
            "text": text,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
        }
        lines_map.setdefault(key, []).append(word_info)

    # On trie les lignes par position verticale moyenne pour garder l'ordre
    sorted_lines = sorted(
        lines_map.values(),
        key=lambda words: sum(w["top"] for w in words) / len(words),
    )
    return sorted_lines


def detect_sensitive_regions(lines: List[List[Dict]]) -> List[Tuple[int, int, int, int]]:
    """
    Retourne une liste de rectangles (x0, y0, x1, y1) à masquer.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    mask_next_address_lines = 0

    for line in lines:
        if not line:
            continue

        line_text = " ".join(w["text"] for w in line)
        line_norm = normalize(line_text)
        line_lower = line_text.lower()

        # --- 1) Cas IBAN : séquence de mots commençant par FR + chiffres ---
        digits_count = 0
        in_iban = False
        for w in line:
            wt = w["text"].replace(" ", "")
            wt_norm = wt.upper()

            if not in_iban:
                # Début d'un IBAN FR typique
                if wt_norm.startswith("FR") and re.search(r"\d", wt_norm):
                    in_iban = True

            if in_iban:
                digits_count += sum(ch.isdigit() for ch in wt_norm)
                x0, y0 = w["left"], w["top"]
                x1 = x0 + w["width"]
                y1 = y0 + w["height"]
                boxes.append((x0, y0, x1, y1))

                # On arrête quand on a suffisamment de chiffres
                if digits_count >= IBAN_MIN_DIGITS:
                    in_iban = False
                    digits_count = 0

        # --- 2) Cas lignes "Nom / Prénom / Client / Titulaire" ---
        has_person_label = any(
            normalize(w["text"]) in PERSON_LABELS for w in line
        )

        has_company_keyword = any(
            normalize(w["text"]) in COMPANY_KEYWORDS for w in line
        )

        if has_person_label and not has_company_keyword:
            # On masque uniquement les mots de type "personne" après les labels
            after_label = False
            for w in line:
                txt_norm = normalize(w["text"])
                if txt_norm in PERSON_LABELS:
                    after_label = True
                    continue  # ne masque pas le label lui-même

                if after_label:
                    if is_person_word(w["text"]):
                        x0, y0 = w["left"], w["top"]
                        x1 = x0 + w["width"]
                        y1 = y0 + w["height"]
                        boxes.append((x0, y0, x1, y1))
                    else:
                        # On s'arrête dès qu'on tombe sur "de / l' / entreprise" etc.
                        break

        # --- 3) Cas ligne "Adresse du client" + quelques lignes suivantes ---
        has_address_label = any(
            normalize(w["text"]) in ADDRESS_LABELS for w in line
        )
        if has_address_label and "banque" not in line_lower:
            # masque tout ce qui suit le mot "Adresse"
            after_addr_label = False
            for w in line:
                txt_norm = normalize(w["text"])
                if txt_norm in ADDRESS_LABELS:
                    after_addr_label = True
                    continue  # ne masque pas le label lui-même
                if after_addr_label:
                    x0, y0 = w["left"], w["top"]
                    x1 = x0 + w["width"]
                    y1 = y0 + w["height"]
                    boxes.append((x0, y0, x1, y1))
            # on demandera de masquer les 2–3 lignes suivantes
            mask_next_address_lines = 3
            continue

        # Si on est dans la "traîne" de l'adresse (lignes suivantes)
        if mask_next_address_lines > 0:
            # Pour limiter les faux positifs, on ne masque que si la ligne "ressemble" à une adresse
            if is_address_like_line(line_text):
                for w in line:
                    x0, y0 = w["left"], w["top"]
                    x1 = x0 + w["width"]
                    y1 = y0 + w["height"]
                    boxes.append((x0, y0, x1, y1))
            mask_next_address_lines -= 1

        # --- 4) (optionnel) Noms isolés sans label ---
        # Si tu veux, tu peux ajouter une passe complémentaire ici
        # pour masquer certains "M. DUPONT Jean" en haut à gauche,
        # mais je la laisse désactivée pour limiter les faux positifs.

    return boxes


def mask_regions_on_image(image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """Dessine des rectangles noirs sur les zones sensibles."""
    # PIL -> OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    # OpenCV -> PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil


def load_pages(file_bytes: bytes, content_type: str) -> List[Image.Image]:
    """Charge un PDF ou une image en liste d'images PIL."""
    if content_type == "application/pdf" or file_bytes[:4] == b"%PDF":
        pages = convert_from_bytes(file_bytes)
        return pages
    # sinon, on suppose image
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [img]


@app.post("/anonymize", summary="Anonymise un PDF/image (noms, adresses, IBAN).")
async def anonymize(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Aucun fichier reçu")

    content = await file.read()
    try:
        pages = load_pages(content, file.content_type or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier: {e}")

    anonymized_pages: List[Image.Image] = []

    for page in pages:
        # OCR
        ocr_data = pytesseract.image_to_data(
            page,
            lang="fra+eng",
            output_type=Output.DICT
        )
        lines = collect_lines(ocr_data)
        boxes = detect_sensitive_regions(lines)
        masked = mask_regions_on_image(page, boxes)
        anonymized_pages.append(masked)

    # Retour en PDF (même si entrée = image)
    pdf_bytes_io = io.BytesIO()
    if len(anonymized_pages) == 1:
        anonymized_pages[0].save(pdf_bytes_io, format="PDF")
    else:
        anonymized_pages[0].save(
            pdf_bytes_io,
            format="PDF",
            save_all=True,
            append_images=anonymized_pages[1:],
        )
    pdf_bytes_io.seek(0)

    return Response(
        content=pdf_bytes_io.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=anonymized.pdf"},
    )
