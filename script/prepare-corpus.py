import re
import sys
from pathlib import Path

from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"
NEW_BOOKS_DIR = BOOKS_DIR / "new"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"

EPUB_FILES = {
    "la-chute": "la-chute.epub",
    "la-peste": "la-peste.epub",
    "l-homme-revolte": "l-homme-revolte.epub",
    "l-etranger": "l-etranger.epub",
}

NEW_EPUB_FILES = {
    "discours-de-suede": "epdf.pub_discours-de-suede.epub",
    "jonas-et-la-pierre-qui-pousse": "epdf.pub_jonas-ou-lartiste-au-travail.epub",
    "l-envers-et-l-endroit": "epdf.pub_lenvers-et-lendroit.epub",
}

TXT_FILES = {
    "le-mythe-de-sisyphe": "sisyphe.txt",
}

METADATA_PATTERNS = [
    r"(?i)classiques des sciences sociales",
    r"(?i)jean-marie tremblay",
    r"(?i)charles bolduc",
    r"(?i)cégep de chicoutimi",
    r"(?i)université du québec",
    r"(?i)bibliothèque paul-émile",
    r"(?i)distributed proofreaders",
    r"(?i)politique\s+d'utilisation",
    r"(?i)reproduction et rediffusion",
    r"(?i)édition électronique",
    r"(?i)domaine public au canada",
    r"(?i)droits d'auteur de votre pays",
    r"(?i)courriel\s*:",
    r"(?i)site web pédagogique",
    r"(?i)page web personnelle",
    r"(?i)à propos de cette édition",
    r"(?i)texte libre de droits",
    r"(?i)conversion informatique",
    r"http[s]?://",
]


def extract_epub_items(epub_path):
    book = epub.read_epub(str(epub_path))
    items = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                items.append(text)
    return items


def is_metadata_section(text):
    matches = sum(1 for p in METADATA_PATTERNS if re.search(p, text))
    return matches >= 2


def is_table_of_contents(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 5 or len(lines) > 60:
        return False
    short_lines = sum(1 for l in lines if len(l) < 50)
    return short_lines / len(lines) > 0.9


def is_front_matter(text):
    stripped = text.replace("\xa0", " ").strip()
    if len(stripped) < 100:
        return True
    if is_metadata_section(stripped):
        return True
    if is_table_of_contents(stripped):
        return True
    front_patterns = [
        r"^OEUVRES\s+D.ALBERT CAMUS",
        r"ŒUVRES\s+D.ALBERT CAMUS",
        r"^REMARQUE",
        r"^©\s*Éditions",
        r"^Éditions Gallimard",
        r"^Tous droits de traduction",
        r"^GALLIMARD\s*$",
        r"^AVANT-PROPOS",
        r"^Table des matières",
        r"^Paru dans Le Livre de Poche",
        r"^Récits.Nouvelles",
        r"^Adaptations et Traductions",
        r"^Cette édition électronique",
        r"^À propos de cette édition",
        r"^Texte libre de droits",
    ]
    for pattern in front_patterns:
        if re.search(pattern, stripped, re.MULTILINE):
            return True
    return False


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\|\s*\d+\s*\|", "", text)

    text = re.sub(
        r"Retour\s+à\s+la\s+table\s+des\s+mat\s*\n?\s*ières",
        "",
        text,
        flags=re.IGNORECASE,
    )

    noise_patterns = [
        r"Classiques des sciences sociales",
        r"Corrections,\s*édition,\s*conversion informatique.*",
        r"Cette édition électronique du livre.*",
        r"©\s*Éditions Gallimard.*",
        r"Éditions Gallimard\n.*",
        r"www\.\S+",
        r"http\S+",
        r"\d+ rue [\w-]+",
        r"\d+ Paris\b",
        r"fondationlaposte",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+$", stripped):
            continue
        if "JMT" in stripped:
            continue
        if "Livre disponible dans" in stripped:
            continue
        if "ISBN" in stripped:
            continue
        if "Code Sodis" in stripped:
            continue
        if "Nord Compo" in stripped:
            continue
        if re.match(r"^Numéro d.édition", stripped):
            continue
        if "a été réalisée le" in stripped and "Gallimard" in stripped:
            continue
        if "repose sur l" in stripped and "édition papier" in stripped:
            continue
        if "Ce document numérique a été réalisé" in stripped:
            continue
        if re.match(r"^Fin du texte$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Note de Transcription$", stripped, re.IGNORECASE):
            break
        if re.match(r"^\[Fin de$", stripped):
            break
        if re.match(r"^BRODARD ET TAUPIN", stripped):
            break
        if re.match(r"^Paris-Coulommiers", stripped):
            break
        cleaned_lines.append(stripped)
    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_epub(epub_path):
    items = extract_epub_items(epub_path)
    content_items = []
    content_started = False

    for item_text in items:
        if not content_started:
            if is_front_matter(item_text):
                continue
            content_started = True
        if is_front_matter(item_text):
            continue
        content_items.append(item_text)

    full_text = "\n\n".join(content_items)
    return clean_text(full_text)


def process_sisyphe(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    start_match = re.search(r"^à Pascal Pia\s*$", text, re.MULTILINE)
    if not start_match:
        start_match = re.search(r"Les pages qui suivent traitent", text)
    if start_match:
        text = text[start_match.start() :]

    footnote_pattern = re.compile(
        r"^\d+\.\s+(?:[A-Z]|Ne manquons|J'ai entendu|Mais non pas|"
        r"À propos|On peut penser|Je n'ai pas|Précisons|"
        r"Même les|A\. —|Il s'agit|La quantité|Même réflexion|"
        r"La volonté|Ce qui importe|Au sens plein|Je pense ici|"
        r"Ce qui est proposé)",
        re.MULTILINE,
    )

    lines = text.split("\n")
    cleaned_lines = []
    in_footnote = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            in_footnote = False
            cleaned_lines.append("")
            continue

        if footnote_pattern.match(stripped):
            in_footnote = True
            continue

        if in_footnote:
            continue

        if re.match(r"^Chapitre \d+$", stripped):
            continue
        if re.match(r"^\d+\.\d+\s+", stripped):
            title = re.sub(r"^\d+\.\d+\s+", "", stripped)
            cleaned_lines.append(title)
            continue
        if re.match(r"^\|\d+", stripped):
            continue
        if re.match(r"^\*$", stripped):
            continue
        if re.match(r"^\d+$", stripped):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    return clean_text(text)


def process_noces_pdf(pdf_path):
    doc = pymupdf.open(str(pdf_path))
    pages_text = []
    for page in doc:
        text = page.get_text()
        text = re.sub(
            r"^\s*Albert Camus, NOCES suivi de L.ÉTÉ\s*\(\d+\)\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(r"^\s*\[\d+\]\s*$", "", text, flags=re.MULTILINE)
        pages_text.append(text)

    full_text = "\n".join(pages_text)

    start_match = re.search(r"NOCES À TIPASA", full_text)
    if start_match:
        full_text = full_text[start_match.start() :]

    end_match = re.search(r"Fin du texte", full_text, re.IGNORECASE)
    if end_match:
        full_text = full_text[: end_match.start()]

    return clean_text(full_text)


def clean_jonas(text):
    start_match = re.search(r"^Gilbert\s*$", text, re.MULTILINE)
    if start_match:
        text = text[start_match.start() :]
    return clean_text(text)


def extract_correspondance_items(epub_path):
    book = epub.read_epub(str(epub_path))
    items = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for notes_div in soup.find_all("div", class_="defnotes"):
                notes_div.decompose()
            for notes_div in soup.find_all("div", class_="ntb"):
                notes_div.decompose()
            for note_ref in soup.find_all("a", class_="apnb"):
                note_ref.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                items.append(text)
    return items


def process_correspondance(epub_path):
    items = extract_correspondance_items(epub_path)
    full_text = "\n\n".join(items)
    full_text = full_text.replace("\xa0", " ")

    letter_pattern = re.compile(
        r"(\d+\s*–\s*(?:ALBERT CAMUS|MARIA CASARÈS)"
        r"\s+À\s+(?:MARIA CASARÈS|ALBERT CAMUS))"
    )
    parts = letter_pattern.split(full_text)

    camus_letters = []
    for i, part in enumerate(parts):
        if "ALBERT CAMUS À MARIA CASARÈS" in part:
            if i + 1 < len(parts):
                letter_body = clean_text(parts[i + 1])
                if letter_body:
                    camus_letters.append(letter_body)

    return "\n\n".join(camus_letters)


def save_text(text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    word_count = len(text.split())
    char_count = len(text)
    print(f"    {output_path.name}: {word_count:,} mots, {char_count:,} caractères")


def main():
    print("=== Préparation du corpus Camus ===\n")

    for name, filename in EPUB_FILES.items():
        path = BOOKS_DIR / filename
        if not path.exists():
            print(f"ERREUR: {path} introuvable", file=sys.stderr)
            sys.exit(1)

    for name, filename in TXT_FILES.items():
        path = BOOKS_DIR / filename
        if not path.exists():
            print(f"ERREUR: {path} introuvable", file=sys.stderr)
            sys.exit(1)

    print("Traitement des EPUBs...\n")

    for name, filename in EPUB_FILES.items():
        print(f"  {name}")
        text = process_epub(BOOKS_DIR / filename)
        save_text(text, CORPUS_DIR / f"{name}.txt")

    print("\n  correspondance (lettres de Camus uniquement)")
    text = process_correspondance(BOOKS_DIR / "correspondance.epub")
    save_text(text, CORPUS_DIR / "correspondance-camus.txt")

    print("\nTraitement des fichiers texte...\n")

    print("  le-mythe-de-sisyphe")
    text = process_sisyphe(BOOKS_DIR / TXT_FILES["le-mythe-de-sisyphe"])
    save_text(text, CORPUS_DIR / "le-mythe-de-sisyphe.txt")

    print("\nTraitement des nouveaux EPUBs...\n")

    for name, filename in NEW_EPUB_FILES.items():
        path = NEW_BOOKS_DIR / filename
        if not path.exists():
            print(f"  SKIP: {path} introuvable", file=sys.stderr)
            continue
        print(f"  {name}")
        text = process_epub(path)
        if name == "jonas-et-la-pierre-qui-pousse":
            text = clean_jonas(text)
        save_text(text, CORPUS_DIR / f"{name}.txt")

    print("\nTraitement du PDF (Noces + L'Été)...\n")

    pdf_files = list(NEW_BOOKS_DIR.glob("*.pdf"))
    if pdf_files:
        print("  noces-et-l-ete")
        text = process_noces_pdf(pdf_files[0])
        save_text(text, CORPUS_DIR / "noces-et-l-ete.txt")
    else:
        print("  SKIP: pas de PDF trouvé")

    print("\n=== Résumé ===\n")
    total_words = 0
    for txt_file in sorted(CORPUS_DIR.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            words = len(f.read().split())
        total_words += words
        print(f"  {txt_file.name}: {words:,} mots")

    print(f"\n  Total: {total_words:,} mots")
    print("\nDone.")


if __name__ == "__main__":
    main()
