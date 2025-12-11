import re
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output
import img2pdf

app = FastAPI()


# --- Détection "simplifiée" de texte sensible ---
def is_sensitive(text: str) -> bool:
    if not text:
        return False
    trimmed = text.strip()

    # Email
    email_regex = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
    if email_regex.search(trimmed):
        return True

    # Téléphone FR très simplifié (10 chiffres commençant par 0)
    phone_regex = re.compile(r"\b0[1-9]\d{8}\b")
    if phone_regex.search(trimmed):
        return True

    # IBAN FR très simplifié
    iban_regex = re.compile(r"\bFR\d{2}[0-9A-Z]{11,}\b", re.I)
    if iban_regex.search(trimmed):
        return True

    # Long numéro (n° client, etc.)
    long_number = re.compile(r"\b\d{8,}\b")
    if long_number.search(trimmed):
        return True

    # Nom / prénom : heuristique très simple
    # Mot qui commence par une majuscule + lettres/accents, longueur >= 3
    name_regex = re.compile(r"^[A-Z][a-zàâäçéèêëîïôöùûüÿ'-]{2,}$")
    if name_regex.match(trimmed):
        return True

    return False


# --- Masque les mots sensibles sur une image PIL ---
def anonymize_image(img: Image.Image) -> Image.Image:
    # OCR : données mot par mot
    # Langue : français + anglais (adaptable)
    data = pytesseract.image_to_data(
        img,
        lang="fra+eng",
        output_type=Output.DICT
    )

    draw = ImageDraw.Draw(img)

    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i]
        if not text or text.strip() == "":
            continue

        if not is_sensitive(text):
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        # Dessine un rectangle noir sur le mot
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            fill="black"
        )

    return img


# --- Anonymise un PDF (bytes) en PDF (bytes) ---
def anonymize_pdf(pdf_bytes: bytes) -> bytes:
    # PDF -> images (une image PIL par page)
    pages = convert_from_bytes(pdf_bytes, dpi=150)  # dpi ajustable

    redacted_images = []
    for idx, page in enumerate(pages, start=1):
        print(f"[anonymize_pdf] Page {idx}/{len(pages)}")
        redacted = anonymize_image(page)
        redacted_images.append(redacted)

    # Images -> PDF
    image_bytes_list = []
    for img in redacted_images:
        buf = BytesIO()
        # On garde en JPEG ou PNG ; ici JPEG pour limiter le poids
        img.save(buf, format="JPEG")
        image_bytes_list.append(buf.getvalue())

    pdf_out = img2pdf.convert(image_bytes_list)
    return pdf_out


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded (field name must be "file")')

    content_type = file.content_type or "application/octet-stream"
    print(f"--- /anonymize called ---")
    print({
        "filename": file.filename,
        "content_type": content_type
    })

    data = await file.read()

    try:
        if content_type == "application/pdf":
            print("Processing as PDF")
            output_bytes = anonymize_pdf(data)
            media_type = "application/pdf"

        elif content_type.startswith("image/"):
            print("Processing as image")
            img = Image.open(BytesIO(data))
            redacted_img = anonymize_image(img)
            buf = BytesIO()
            # On renvoie en PNG pour être simple
            redacted_img.save(buf, format="PNG")
            output_bytes = buf.getvalue()
            media_type = "image/png"

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported mimetype: {content_type}"
            )

        return Response(content=output_bytes, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /anonymize:", e)
        raise HTTPException(status_code=500, detail=str(e))
