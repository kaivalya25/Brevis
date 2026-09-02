"""
build_dataset.py - optional, offline only.

This never runs on the phone. It runs once on a laptop to curate a bigger
books.json than the list hard-coded in index.html, using the UCSD Goodreads
dump plus Gemini embeddings and a local Chroma collection.

ChromaDB is a build-time tool here. Querying it at runtime would need a
server, which would break the "costs nothing, hosts nowhere" rule Brevis
is built around.

Inputs (download from
https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html and leave gzipped
in this folder):

    goodreads_books.json.gz
    goodreads_book_authors.json.gz

Output:

    books.json   - the same {t,a,r,y,g,d} shape index.html already uses.

Setup:

    pip install chromadb google-generativeai
    export GOOGLE_API_KEY=...        (set GOOGLE_API_KEY=... on Windows)
    python build_dataset.py
"""

import gzip
import json
import os
import sys
import time

import chromadb
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Settings. Everything worth tuning is here.
# ---------------------------------------------------------------------------

BOOKS_GZ = "goodreads_books.json.gz"
AUTHORS_GZ = "goodreads_book_authors.json.gz"
OUT = "books.json"

MIN_RATINGS = 50_000      # how well known a book has to be
MIN_RATING = 3.5          # how well liked
MIN_DESC = 200            # characters of blurb needed to embed anything useful

EMBED_MODEL = "models/text-embedding-004"
BATCH = 100               # descriptions per embedding call
PAUSE = 1.0               # seconds between calls, to stay inside the free tier
PER_MOOD = 30             # how many books each mood pulls out of Chroma

# The moods, and the phrase each one is matched against. The tags on the left
# are the same strings index.html uses in the g:[] list of every book.
MOODS = {
    "cozy": "a warm, gentle, comforting story in a small community, low stakes and kind",
    "thrilling": "a tense page-turning thriller full of suspense, danger and pursuit",
    "romantic": "a love story about longing, attraction and a relationship at its centre",
    "sci-fi": "science fiction about technology, space travel, the future or artificial minds",
    "fantasy": "epic fantasy with magic, invented worlds, myth and quests",
    "melancholic": "a quiet, sad, reflective book about loss, memory and regret",
    "funny": "a comic novel that is witty, absurd and laugh-out-loud funny",
    "mind-bending": "a strange, surreal, philosophical book that plays with reality and perception",
    "historical": "historical fiction set in a carefully researched past era",
    "dark": "a bleak, violent, disturbing book with a grim atmosphere",
    "classic": "a canonical literary classic taught and reread for generations",
}


# ---------------------------------------------------------------------------
# 1. Read the dumps. Both files are streamed a line at a time - the books file
#    holds about 2.3 million records and will not fit in memory.
# ---------------------------------------------------------------------------

def load_authors():
    names = {}
    with gzip.open(AUTHORS_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            names[row["author_id"]] = row["name"]
    return names


def first_sentence(text):
    """The hook shown under each title: the blurb's first sentence, trimmed."""
    text = " ".join(text.split())
    for stop in (". ", "! ", "? "):
        i = text.find(stop)
        if 20 < i < 200:
            return text[: i + 1].strip()
    return (text[:180] + "...") if len(text) > 180 else text


def load_books(author_names):
    kept = []
    with gzip.open(BOOKS_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)

            try:
                count = int(row.get("ratings_count") or 0)
                rating = float(row.get("average_rating") or 0)
            except ValueError:
                continue

            if count < MIN_RATINGS or rating < MIN_RATING:
                continue
            if row.get("language_code") not in ("eng", "en-US", "en-GB", "en-CA"):
                continue

            desc = (row.get("description") or "").strip()
            if len(desc) < MIN_DESC:
                continue

            authors = row.get("authors") or []
            if not authors:
                continue
            author = author_names.get(authors[0].get("author_id"))
            if not author:
                continue

            try:
                year = int(row.get("original_publication_year") or row.get("publication_year") or 0)
            except ValueError:
                year = 0

            title = (row.get("title") or "").strip()
            if not title:
                continue

            kept.append({
                "t": title,
                "a": author,
                "r": round(rating, 2),
                "y": year,
                "d": first_sentence(desc),
                "_desc": desc,
            })
    return kept


# ---------------------------------------------------------------------------
# 2. Embed the descriptions in batches, with a pause between calls.
# ---------------------------------------------------------------------------

def embed(texts, task):
    out = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        res = genai.embed_content(model=EMBED_MODEL, content=chunk, task_type=task)
        out.extend(res["embedding"])
        print("  embedded %d/%d" % (min(i + BATCH, len(texts)), len(texts)))
        time.sleep(PAUSE)
    return out


# ---------------------------------------------------------------------------
# 3. Load Chroma, query once per mood, dedupe, write books.json.
# ---------------------------------------------------------------------------

def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GOOGLE_API_KEY in the environment first.")
    for path in (BOOKS_GZ, AUTHORS_GZ):
        if not os.path.exists(path):
            sys.exit("Missing %s - download it from the UCSD Goodreads page." % path)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    print("Reading authors...")
    author_names = load_authors()

    print("Reading books...")
    books = load_books(author_names)
    print("%d books passed the filters." % len(books))
    if not books:
        sys.exit("Nothing survived the filters - loosen MIN_RATINGS or MIN_RATING.")

    print("Embedding descriptions...")
    vectors = embed([b["_desc"] for b in books], "retrieval_document")

    print("Loading Chroma...")
    client = chromadb.Client()
    col = client.create_collection("brevis")
    col.add(
        ids=[str(i) for i in range(len(books))],
        embeddings=vectors,
        metadatas=[{"i": i} for i in range(len(books))],
    )

    print("Querying one mood at a time...")
    picked = {}          # title -> book, so one book can carry several moods
    mood_vectors = embed(list(MOODS.values()), "retrieval_query")

    for (tag, _phrase), vec in zip(MOODS.items(), mood_vectors):
        hits = col.query(query_embeddings=[vec], n_results=PER_MOOD)
        for meta in hits["metadatas"][0]:
            book = books[meta["i"]]
            entry = picked.setdefault(book["t"], {
                "t": book["t"], "a": book["a"], "r": book["r"],
                "y": book["y"], "g": [], "d": book["d"],
            })
            if tag not in entry["g"]:
                entry["g"].append(tag)
        print("  %-13s %d kept so far" % (tag, len(picked)))

    out = sorted(picked.values(), key=lambda b: -b["r"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print("Wrote %s with %d books." % (OUT, len(out)))


if __name__ == "__main__":
    main()
