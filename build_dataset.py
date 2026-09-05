"""
build_dataset.py - the offline half of Brevis. Runs on a laptop, never on a phone.

WHAT IT DOES, IN ONE BREATH
    Reads every book in a multi-million-row Goodreads dump, ranks them with a
    recommendation score, embeds the survivors into a local vector database,
    sorts them into moods, works out which books are near neighbours of which,
    and writes a small books.json that index.html can serve straight off a
    static host.

WHY THE HEAVY LIFTING HAPPENS HERE
    Brevis has no backend. A vector database at runtime would need a server,
    which would break the "costs nothing to run" rule. So all the retrieval
    work is done once, offline, and only the *result* ships to the phone.

TWO SOURCES, PICK ONE

    --source hf       (default)  No download. Streams the parquet copy of a
                                 3.9-million-row Goodreads dataset straight off
                                 Hugging Face over HTTPS, pulling only the seven
                                 columns we actually use and throwing each batch
                                 away after scoring it. Nothing is kept on disk.

    --source local               The classic UCSD Book Graph dump, if you have
                                 already downloaded goodreads_books.json.gz and
                                 goodreads_book_authors.json.gz (about 2.3m
                                 books, ~2.2 GB). Streamed line by line.

RUNNING IT

    pip install chromadb pyarrow fsspec huggingface_hub numpy
    python build_dataset.py                      # scan everything
    python build_dataset.py --scan-limit 100000  # a quick trial first
    python build_dataset.py --no-vectors         # skip the Gemini stage

    A GOOGLE_API_KEY is needed ONLY for the last stage, which embeds the few
    thousand exported books so the app can do semantic search in the browser.
    Everything before that runs offline and free.
"""

import argparse
import gzip
import heapq
import json
import os
import re
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# SETTINGS - the knobs worth turning are all here.
# ---------------------------------------------------------------------------

HF_DATASET = "BrightData/Goodreads-Books"
HF_PARQUET_INDEX = "https://datasets-server.huggingface.co/parquet?dataset=" + HF_DATASET.replace("/", "%2F")

LOCAL_BOOKS_GZ = "goodreads_books.json.gz"
LOCAL_AUTHORS_GZ = "goodreads_book_authors.json.gz"

MIN_DESC = 200          # a blurb shorter than this cannot be embedded usefully
MIN_VOTES = 25          # a book with a handful of ratings tells us nothing
PRIOR_VOTES = 2500      # "m" in the weighted rating below - see score_book()

EMBED_MODEL = "models/text-embedding-004"   # used only in the last stage
EMBED_DIMS = 128        # the browser downloads one vector per book, so keep it small
GEMINI_BATCH = 100
GEMINI_PAUSE = 0.6      # seconds between calls, to stay inside the free tier

CHROMA_BATCH = 5000     # Chroma refuses much larger single adds

# The moods, and the sentence each one is matched against in vector space.
# The tags on the left are the same strings index.html uses in every book's g[].
MOODS = {
    "cozy": "a warm, gentle, comforting story in a small community, kind and low stakes",
    "thrilling": "a tense page-turning thriller full of suspense, danger, pursuit and secrets",
    "romantic": "a love story about longing and attraction, with a relationship at its centre",
    "sci-fi": "science fiction about space travel, future technology, aliens or artificial minds",
    "fantasy": "epic fantasy with magic, invented worlds, myth, dragons and quests",
    "melancholic": "a quiet, sad, reflective novel about loss, memory, grief and regret",
    "funny": "a comic novel that is witty, absurd, satirical and laugh-out-loud funny",
    "mind-bending": "a strange, surreal, philosophical book that plays with reality, time and perception",
    "historical": "historical fiction set in a carefully researched past era or wartime",
    "dark": "a bleak, violent, disturbing novel with a grim and unsettling atmosphere",
    "classic": "a canonical work of literature, taught in schools and reread for generations",
}


# ---------------------------------------------------------------------------
# STAGE 1 - READING THE DUMP
#
# Both readers yield the same plain dictionary, so nothing downstream has to
# care which source was used.
# ---------------------------------------------------------------------------

YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-9]{2})")


def load_env(path=".env"):
    """Read KEY=value lines out of a .env file into the environment.

    Only build_dataset.py uses this. The app itself cannot: it runs in a
    browser with no server behind it, so any file it could read, a visitor
    could read too. That is why the app asks each reader for their own key
    and keeps it in their browser instead.

    A real environment variable always wins over the file.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value


def parse_year(value):
    """Years arrive as ints, as '1991', and as 'First published May 3, 1991'."""
    if value is None:
        return 0
    m = YEAR_RE.search(str(value))
    return int(m.group(1)) if m else 0


def clean(text):
    return " ".join(str(text or "").split())


def parse_list(value):
    """Several columns arrive as a string that looks like a list, e.g.
    '["Dennis Fischer"]' or "['Fiction', 'Horror']". Return the real list."""
    text = clean(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [clean(x) for x in parsed if clean(x)]
        except ValueError:
            pass
        text = text.strip("[]")
    return [p for p in (clean(x).strip("\"'") for x in text.split(",")) if p]


def first_author(value):
    names = parse_list(value)
    return names[0] if names else clean(value)


# This dataset has no language column, and it is full of translated editions -
# the German and Italian Game of Thrones sit right beside the English one. An
# app that writes English summaries should not offer them, so blurbs are
# checked for the small words English cannot write a paragraph without. Real
# English prose is roughly a fifth of these; other languages score near zero.
ENGLISH_HINTS = frozenset("""
a an the and or but of to in on at for with from by as is was are were be been
that this these those it its he she his her they them their you your not no
who what when where which while has have had will would can could about into
""".split())

WORD_RE = re.compile(r"[a-zA-Z']+")


def looks_english(text):
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < 30:
        return False
    hits = sum(1 for w in words if w in ENGLISH_HINTS)
    return (hits / len(words)) >= 0.18


# Brevis summarises plots, so it wants novels. Left alone, the ranking fills up
# with Bibles, cookbooks and art-history monographs, because those are popular
# and well rated. The genres column is well populated, so use it: a book has to
# claim at least one fiction label, must not also claim to be nonfiction, and
# must not be a form that has no single plot to tell.
FICTION_LABELS = frozenset([
    "fiction", "fantasy", "science fiction", "mystery", "romance", "thriller",
    "historical fiction", "horror", "contemporary", "literary fiction", "crime",
    "adventure", "young adult", "classics", "dystopia", "paranormal", "suspense",
    "magical realism", "urban fantasy", "epic fantasy", "novels", "chick lit",
    "contemporary romance", "womens fiction", "gothic",
])

BLOCKED_LABELS = frozenset([
    "nonfiction", "non fiction", "reference", "textbooks", "picture books",
    "comics", "graphic novels", "manga", "poetry", "short stories",
    "anthologies", "essays", "self help", "cookbooks", "religion",
    "music", "art", "biography", "memoir", "travel", "sports",
])


# Even among real novels the dump carries things that are not one story: free
# samplers, boxed sets, omnibus editions and split volumes. There is nothing
# for the summariser to summarise in "Words of Radiance, Part 2".
TITLE_JUNK = re.compile(
    r"\b(sampler|omnibus|excerpt|preview|teaser|boxed set|box set|bundle)\b"
    r"|\bcollection\b"
    # Unofficial cash-ins: study guides, "-- Review", "Summary & Analysis of".
    r"|\b(review|summary|study ?guide|analysis|companion|quiz|trivia"
    r"|sparknotes|cliffsnotes|unofficial|key takeaways|conversation starters)\b"
    # Tie-in merchandise rather than a story. Kept deliberately narrow: a
    # pattern as loose as "guide to" would throw out The Hitchhiker's Guide
    # to the Galaxy, which is very much a novel.
    r"|\b(colou?ring book|activity book|sticker book|workbook"
    r"|interactive adventure|choose your own)\b"
    # "Skulduggery Pleasant #1-9", "The All Souls Trilogy"
    r"|#\s*\d+\s*[-–—]\s*\d+"
    r"|\b(trilogy|quartet|quintet|duology|tetralogy)\b"
    # "The New Annotated Sherlock Holmes: The Complete Short Stories"
    r"|\bannotated\b"
    r"|\bcomplete\b.{0,20}\b(short stories|stories|works|tales)\b"
    r"|\bpart\s+(one|two|three|four|1|2|3|4)\b"
    r"|\bbooks?\s*\d+\s*[-–]\s*\d+\b"
    r"|\bcomplete\b.{0,30}\b(trilogy|series|saga|novels)\b",
    re.IGNORECASE)

# An omnibus usually names its contents after a colon: "The Restoration
# Collection: Last Light, Night Light, True Light, Dawn's Light". Two or more
# commas after a colon is almost always that. It does occasionally catch a
# real novel with a comma-heavy subtitle, which is a fair trade when there are
# millions of candidates and only a few thousand places to fill.
TITLE_LIST = re.compile(r":[^:]*,[^,]*,")


# Some bundles have a perfectly innocent title - "The Giver Quartet", "The
# Books of the South" - and only give themselves away in the blurb, which
# cheerfully says "all three novels" or "books 1-9 in one volume".
BUNDLE_BLURB = re.compile(
    r"\bbooks?\s*\d+\s*[-–—]\s*\d+\b"
    r"|\b(omnibus|boxed set|box set)\b"
    r"|\ball (two|three|four|five|six|seven|eight|nine|ten) (books|novels)\b"
    r"|\bcomplete series\b"
    r"|\bin one volume\b"
    # Bundles love to list their contents: "Includes: A / B / C"
    r"|\bincludes:\s*[^/]{3,60}/[^/]{3,60}/",
    re.IGNORECASE)


def is_single_story(title, description=""):
    if TITLE_JUNK.search(title) or TITLE_LIST.search(title):
        return False
    return not BUNDLE_BLURB.search(description)


def is_novel(shelves):
    labels = {s.strip().lower() for s in shelves.split(",") if s.strip()}
    if not labels:
        return False
    if labels & BLOCKED_LABELS:
        return False
    return bool(labels & FICTION_LABELS)


def read_hf(scan_limit):
    """Stream the parquet copy off Hugging Face. Nothing touches the disk.

    Parquet is columnar, so asking for seven columns means only those columns
    cross the network - a fraction of the full dataset - and each batch is
    discarded as soon as it has been scored.
    """
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("Streaming needs pyarrow and fsspec:  pip install pyarrow fsspec")

    print("Asking Hugging Face which parquet files make up %s ..." % HF_DATASET)
    with urllib.request.urlopen(HF_PARQUET_INDEX, timeout=60) as resp:
        files = json.load(resp)["parquet_files"]
    urls = [f["url"] for f in files]
    print("  %d parquet files to stream." % len(urls))

    columns = ["name", "author", "star_rating", "num_ratings",
               "summary", "genres", "first_published"]
    fs = fsspec.filesystem("https")
    seen = 0

    for n, url in enumerate(urls, 1):
        print("  file %d/%d" % (n, len(urls)))
        with fs.open(url) as handle:
            pf = pq.ParquetFile(handle)
            for batch in pf.iter_batches(batch_size=20000, columns=columns):
                rows = batch.to_pylist()
                for row in rows:
                    seen += 1
                    yield {
                        "t": clean(row.get("name")),
                        "a": first_author(row.get("author")),
                        "r": float(row.get("star_rating") or 0.0),
                        "n": int(row.get("num_ratings") or 0),
                        "y": parse_year(row.get("first_published")),
                        "d": clean(row.get("summary")),
                        "shelves": ", ".join(parse_list(row.get("genres"))[:6]),
                    }
                    if scan_limit and seen >= scan_limit:
                        return


def read_local(scan_limit):
    """The original UCSD dump: two gzipped files, streamed a line at a time."""
    for path in (LOCAL_BOOKS_GZ, LOCAL_AUTHORS_GZ):
        if not os.path.exists(path):
            sys.exit("Missing %s. Download it from the UCSD Book Graph page, "
                     "or drop --source local to stream instead." % path)

    print("Reading author names ...")
    names = {}
    with gzip.open(LOCAL_AUTHORS_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            names[row["author_id"]] = row["name"]
    print("  %d authors." % len(names))

    seen = 0
    with gzip.open(LOCAL_BOOKS_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            seen += 1
            row = json.loads(line)
            authors = row.get("authors") or []
            author = names.get(authors[0]["author_id"], "") if authors else ""
            try:
                rating = float(row.get("average_rating") or 0)
                votes = int(row.get("ratings_count") or 0)
            except ValueError:
                continue
            yield {
                "t": clean(row.get("title")),
                "a": clean(author),
                "r": rating,
                "n": votes,
                "y": parse_year(row.get("original_publication_year") or row.get("publication_year")),
                "d": clean(row.get("description")),
                "shelves": "",
            }
            if scan_limit and seen >= scan_limit:
                return


# ---------------------------------------------------------------------------
# STAGE 2 - THE RECOMMENDATION SCORE
#
# A raw average is a bad ranking: an obscure book with nine five-star ratings
# would beat every classic ever written. The standard fix is a weighted rating,
# which drags a book's average toward the global average until it has earned
# enough votes to speak for itself:
#
#     score = (v / (v + m)) * R  +  (m / (v + m)) * C
#
#     R = this book's average rating
#     v = how many people rated it
#     C = the average rating across the whole dump
#     m = PRIOR_VOTES, how many votes it takes to be trusted
#
# C is tracked as a running mean while the single pass is underway, which is
# accurate long before the scan is over, and every shortlisted book is rescored
# with the final C at the end.
# ---------------------------------------------------------------------------

def score_book(rating, votes, mean_rating):
    return (votes / (votes + PRIOR_VOTES)) * rating + (PRIOR_VOTES / (votes + PRIOR_VOTES)) * mean_rating


def scan(records, shortlist_size):
    """Read every record; keep only the best `shortlist_size` of them.

    A heap holds the running best, so memory stays flat no matter how many
    millions of rows go past.
    """
    heap = []           # min-heap of (score, tiebreak, record)
    tiebreak = 0
    total = 0
    usable = 0
    rating_sum = 0.0
    rating_n = 0
    mean_rating = 3.8   # a sane starting guess, replaced within a few thousand rows

    start = time.time()
    for rec in records:
        total += 1

        if rec["n"] > 0 and 0 < rec["r"] <= 5:
            rating_sum += rec["r"]
            rating_n += 1
            mean_rating = rating_sum / rating_n

        if total % 200000 == 0:
            print("    scanned %s rows, kept %d, mean rating %.3f, %.0fs elapsed"
                  % ("{:,}".format(total), len(heap), mean_rating, time.time() - start))

        if not rec["t"] or not rec["a"]:
            continue
        if rec["n"] < MIN_VOTES or not (0 < rec["r"] <= 5):
            continue
        if len(rec["d"]) < MIN_DESC:
            continue
        if not is_novel(rec["shelves"]):
            continue
        if not is_single_story(rec["t"], rec["d"]):
            continue
        if not looks_english(rec["d"]):
            continue

        usable += 1
        s = score_book(rec["r"], rec["n"], mean_rating)
        tiebreak += 1

        if len(heap) < shortlist_size:
            heapq.heappush(heap, (s, tiebreak, rec))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, tiebreak, rec))

    print("  scanned %s rows in %.0fs." % ("{:,}".format(total), time.time() - start))
    print("  %s had a real blurb, a rating and enough votes." % "{:,}".format(usable))
    print("  global mean rating: %.3f" % mean_rating)

    # Rescore the shortlist with the final mean, now that we know it.
    books = [rec for _, _, rec in heap]
    for b in books:
        b["score"] = score_book(b["r"], b["n"], mean_rating)
    books.sort(key=lambda b: -b["score"])
    return books


# ---------------------------------------------------------------------------
# STAGE 3 - THE VECTOR DATABASE
#
# Chroma embeds the blurbs with a small model that runs locally on the CPU.
# No API key, no network, no cost. This is the retrieval index that everything
# below is built on.
# ---------------------------------------------------------------------------

def build_index(books):
    import chromadb

    client = chromadb.EphemeralClient()
    col = client.create_collection("brevis")

    print("Embedding %d blurbs locally (this is the slow part) ..." % len(books))
    start = time.time()
    for i in range(0, len(books), CHROMA_BATCH):
        chunk = books[i:i + CHROMA_BATCH]
        col.add(
            ids=[str(i + j) for j in range(len(chunk))],
            documents=[b["d"][:2000] for b in chunk],
            metadatas=[{"i": i + j} for j in range(len(chunk))],
        )
        print("    %d/%d embedded, %.0fs elapsed" % (min(i + CHROMA_BATCH, len(books)), len(books), time.time() - start))
    return col


def tag_moods(col, books, per_mood):
    """One vector search per mood. A book can come back for several moods,
    which is how it ends up with more than one tag."""
    print("Sorting books into moods ...")
    for tag, phrase in MOODS.items():
        hits = col.query(query_texts=[phrase], n_results=per_mood)
        for meta in hits["metadatas"][0]:
            book = books[meta["i"]]
            book.setdefault("g", [])
            if tag not in book["g"]:
                book["g"].append(tag)
        print("    %-13s %d books" % (tag, per_mood))


def top_up_authors(col, books, min_per_author):
    """Rescue authors the mood queries left stranded.

    A mood query returns the best few hundred books for that mood, and those
    are scattered across thousands of authors. An author can easily come back
    with one tagged book while three more of theirs sit in the shortlist
    untagged - and an author with one book is a dead end in the app, so they
    would be dropped entirely.

    So: for any author who is close to qualifying, take their untagged books
    and give each one the single mood it sits nearest to. Same vectors, same
    mood sentences, just asked the other way round - which mood is closest to
    this book, rather than which books are closest to this mood.
    """
    import numpy as np
    from chromadb.utils import embedding_functions

    by_author = {}
    for b in books:
        by_author.setdefault(b["a"], []).append(b)

    need = []
    for group in by_author.values():
        tagged = [b for b in group if b.get("g")]
        if tagged and len(group) >= min_per_author and len(tagged) < min_per_author:
            need.extend([b for b in group if not b.get("g")])

    if not need:
        print("  no authors needed topping up.")
        return

    mood_tags = list(MOODS.keys())
    ef = embedding_functions.DefaultEmbeddingFunction()
    mvec = np.array(ef(list(MOODS.values())), dtype="float32")
    mvec /= (np.linalg.norm(mvec, axis=1, keepdims=True) + 1e-9)

    got = col.get(ids=[str(b["_row"]) for b in need], include=["embeddings"])
    order = {int(i): p for p, i in enumerate(got["ids"])}
    vecs = np.array(got["embeddings"], dtype="float32")
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    for b in need:
        b["g"] = [mood_tags[int(np.argmax(mvec @ vecs[order[b["_row"]]]))]]

    print("  topped up %d books so their authors stay reachable." % len(need))


# ---------------------------------------------------------------------------
# STAGE 4 - CHOOSING WHAT SHIPS
#
# Only books that landed in at least one mood are worth shipping. On top of
# that the app has one structural requirement: every author it shows must have
# at least three books, or tapping that author leads to a dead end.
# ---------------------------------------------------------------------------

def choose(books, export_max, min_per_author, max_per_author):
    tagged = [b for b in books if b.get("g")]
    print("  %d books picked up at least one mood." % len(tagged))

    # Drop exact duplicates - the dump has many editions of the same title.
    seen = set()
    unique = []
    for b in sorted(tagged, key=lambda b: -b["score"]):
        key = (b["t"].lower(), b["a"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)
    print("  %d after removing duplicate editions." % len(unique))

    # Keep whole authors, best authors first, until we hit the cap. Trimming to
    # the cap first and pruning afterwards would strand authors on one or two
    # books, which is exactly the dead end we are avoiding.
    by_author = {}
    for b in unique:
        by_author.setdefault(b["a"], []).append(b)

    ranked = sorted(
        (a for a in by_author.values() if len(a) >= min_per_author),
        key=lambda group: -max(b["score"] for b in group),
    )
    print("  %d authors have %d or more books." % (len(ranked), min_per_author))

    # The app never shows more than five books by one author, so letting a
    # prolific series writer take twenty slots just crowds other authors out.
    out = []
    for group in ranked:
        best = sorted(group, key=lambda b: -b["score"])[:max_per_author]
        if len(out) + len(best) > export_max:
            continue
        out.extend(best)
    print("  exporting %d books by %d authors."
          % (len(out), len({b["a"] for b in out})))
    return out


def add_neighbours(col, chosen, how_many=6):
    """The recommendation half: for each exported book, its nearest neighbours
    in vector space, minus anything by the same author. This is what powers
    'if you liked this' on the summary screen, and it is also the retrieval
    step the summary prompt is grounded in."""
    import numpy as np

    print("Working out which books are like which ...")
    got = col.get(ids=[str(b["_row"]) for b in chosen], include=["embeddings"])

    # col.get does not promise to preserve order, so map ids back to positions.
    order = {int(i): p for p, i in enumerate(got["ids"])}
    vecs = np.array(got["embeddings"], dtype="float32")
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    rows = np.array([order[b["_row"]] for b in chosen])
    matrix = vecs[rows]
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, -1.0)

    authors = [b["a"] for b in chosen]
    for i, book in enumerate(chosen):
        ranking = np.argsort(-sim[i])[: how_many * 4]
        picks = [int(j) for j in ranking if authors[j] != authors[i]][:how_many]
        book["s"] = picks


# ---------------------------------------------------------------------------
# STAGE 5 - VECTORS FOR THE BROWSER (optional, needs GOOGLE_API_KEY)
#
# The local model from stage 3 cannot run in a phone browser, so semantic
# search in the app uses Gemini instead: the page embeds whatever the user
# typed, and compares it against these vectors, which were made by the same
# model at the same size. A few thousand books is around thirty API calls.
# ---------------------------------------------------------------------------

def build_vectors(chosen, out_path):
    import numpy as np

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("No GOOGLE_API_KEY set - skipping browser vectors. "
              "Search will fall back to matching on names.")
        return False

    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai is not installed - skipping browser vectors.")
        return False

    genai.configure(api_key=key)
    texts = ["%s by %s. %s" % (b["t"], b["a"], b["d"][:600]) for b in chosen]

    print("Embedding %d books with Gemini for in-browser search ..." % len(texts))
    vectors = []
    for i in range(0, len(texts), GEMINI_BATCH):
        chunk = texts[i:i + GEMINI_BATCH]
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=chunk,
            task_type="retrieval_document",
            output_dimensionality=EMBED_DIMS,
        )
        vectors.extend(res["embedding"])
        print("    %d/%d" % (min(i + GEMINI_BATCH, len(texts)), len(texts)))
        time.sleep(GEMINI_PAUSE)

    v = np.array(vectors, dtype="float32")
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # Quantise to one byte per number. 3000 books at 128 dimensions is under
    # 400 KB, which is a reasonable thing to ask a phone to download.
    q = np.clip(np.round(v * 127.0), -127, 127).astype("int8")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "model": EMBED_MODEL,
            "dims": EMBED_DIMS,
            "scale": 1.0 / 127.0,
            "data": [row.tolist() for row in q],
        }, fh, separators=(",", ":"))
    print("  wrote %s" % out_path)
    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build the Brevis dataset.")
    ap.add_argument("--source", choices=["hf", "local"], default="hf",
                    help="hf streams from Hugging Face (no download); local reads the UCSD .gz files")
    ap.add_argument("--scan-limit", type=int, default=0,
                    help="stop after this many rows - 0 means read everything")
    ap.add_argument("--shortlist", type=int, default=60000,
                    help="how many top-scoring books to embed")
    ap.add_argument("--export", type=int, default=3000,
                    help="how many books end up in books.json")
    ap.add_argument("--per-mood", type=int, default=400,
                    help="books pulled out of the index per mood")
    ap.add_argument("--min-per-author", type=int, default=3,
                    help="authors with fewer books than this are dropped")
    ap.add_argument("--max-per-author", type=int, default=6,
                    help="how many books one author may contribute")
    ap.add_argument("--no-vectors", action="store_true",
                    help="skip the Gemini stage even if a key is set")
    ap.add_argument("--out", default="books.json")
    ap.add_argument("--vectors-out", default="vectors.json")
    args = ap.parse_args()

    load_env()

    print("=" * 66)
    print("BREVIS DATASET BUILD")
    print("=" * 66)

    records = read_hf(args.scan_limit) if args.source == "hf" else read_local(args.scan_limit)

    print("\n[1/5] Scanning the dump ...")
    books = scan(records, args.shortlist)
    if not books:
        sys.exit("Nothing survived the filters. Lower MIN_VOTES or MIN_DESC.")
    print("  shortlist: %d books, best score %.3f, worst %.3f"
          % (len(books), books[0]["score"], books[-1]["score"]))

    for i, b in enumerate(books):
        b["_row"] = i

    print("\n[2/5] Building the vector index ...")
    col = build_index(books)

    print("\n[3/5] Retrieval by mood ...")
    tag_moods(col, books, args.per_mood)
    top_up_authors(col, books, args.min_per_author)

    print("\n[4/5] Choosing what ships ...")
    chosen = choose(books, args.export, args.min_per_author, args.max_per_author)
    if not chosen:
        sys.exit("Nothing to export. Try a bigger --shortlist or --per-mood.")
    add_neighbours(col, chosen)

    print("\n[5/5] Writing files ...")
    out = []
    for b in chosen:
        out.append({
            "t": b["t"],
            "a": b["a"],
            "r": round(b["r"], 2),
            "y": b["y"],
            "g": b["g"],
            "d": first_sentence(b["d"]),
            "x": b["d"][:700],      # the retrieval snippet the summary is grounded in
            "k": b["shelves"],      # reader shelf labels, more grounding
            "s": b["s"],            # indices of similar books, for recommendations
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(args.out) / 1024.0
    print("  wrote %s - %d books, %.0f KB" % (args.out, len(out), size))

    if not args.no_vectors:
        build_vectors(chosen, args.vectors_out)

    print("\nDone. Put books.json (and vectors.json) next to index.html.")


def first_sentence(text):
    """The one-line hook shown under each title."""
    text = clean(text)
    for stop in (". ", "! ", "? "):
        i = text.find(stop)
        if 20 < i < 200:
            return text[: i + 1].strip()
    return (text[:180] + "...") if len(text) > 180 else text


if __name__ == "__main__":
    main()
