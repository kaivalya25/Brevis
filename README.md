# Brevis

Worth your week, or worth skipping? Brevis gives you two summaries of a novel —
one short, one long — with enough detail to decide, and never enough to spoil it.

One HTML file. No server, no database, no build step, no packages, no accounts,
no tracking. It works when you double-click it and it works on a static host.

**Live:** https://kaivalya25.github.io/Brevis/

---

## How it works

Three ideas, and the interesting part is that all three survive having no backend.

### 1. A recommendation system

`build_dataset.py` reads a Goodreads dump of several million books and ranks
every one of them with a weighted rating — the standard fix for the problem that
a book with nine five-star reviews would otherwise outrank every classic ever
written:

```
score = (v / (v + m)) · R  +  (m / (v + m)) · C
```

`R` is the book's own average, `v` how many people rated it, `C` the average
across the whole dump, and `m` how many votes it takes before a book is trusted
to speak for itself. A heap keeps only the best few thousand as the millions
stream past, so memory stays flat.

Inside the app, authors are ranked again — averaged within the mood you picked,
with a nudge for having written several good books in it rather than one lucky one.

### 2. Retrieval-augmented generation

The shortlisted blurbs are embedded into a local Chroma vector database. Each of
the eleven moods is a sentence — *"a quiet, sad, reflective novel about loss,
memory, grief and regret"* — and querying the index with it is what sorts books
into moods. The same index gives every book its nearest neighbours, which is
where **If you liked this** comes from.

None of that runs on your phone. A vector database at runtime would need a
server, so retrieval happens once offline and only the result ships.

At runtime the app still retrieves before it generates: opening a summary
gathers the book's blurb, rating, moods, shelves and nearest neighbours out of
the local dataset and puts them in the prompt as grounding, instructing the model
to trust that material over its own memory.

The search box on the mood screen is genuine vector search in the browser. Your
phrase goes to Gemini's embedding model, comes back as 128 numbers, and is
compared against one stored vector per book — same model, same size, same space,
so a dot product ranks them. The vectors are quantised to a byte each, which is
why the index is under a megabyte rather than several.

### 3. Two summaries, and no spoilers

Every book offers two lengths, and **neither is fetched until you tap for it**:

- **Short** — 200–300 words. Enough to tell in a minute whether it's for you.
- **Long** — 700–1000 words. The setup, the characters and what they want, the
  themes, the tone and pace of the writing, and who would love it.

The prompts spend more words on what *not* to say than on what to write: no
ending, no twist, no hint that a twist is coming, nothing past roughly the first
quarter of the book. If a fact would spoil the read it is left out entirely
rather than hedged. The instruction is to describe the promise of a book, not
its payoff.

You can browse the whole catalogue — every mood, every author, every book — and
Gemini is never called once. Each length is cached per book, so switching
between them is instant after the first time.

The one place Gemini acts on its own initiative is when you name a book or author
the catalogue doesn't have. Search for something missing and Brevis offers to
look it up; the same two lengths then apply, written from scratch. That offer is
always a tap, never automatic.

---

## The API key

There is no settings screen. The first time you ask for a summary, the app asks
for a Gemini key inline, stores it in your browser, and never asks again.

**Why it can't just ship with one.** Brevis is a static page with no server. Any
file the page can read, a visitor can read — so a key committed to this
repository would be public, and a public Gemini key is scraped and abused within
hours. There is nowhere on a static host to hide a secret. Each reader brings
their own, and it stays on their device.

Building the dataset is different, because that runs on your laptop:

|  | Where the key lives | Why |
|---|---|---|
| **`build_dataset.py`** | `.env` on your machine | Runs locally. `.env` is gitignored. |
| **The app** | Typed in once by each reader, kept in their browser | No server. A key in a static site is a public key. |

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### Models

Both model names are aliases rather than pinned versions, deliberately:

- Writing — `models/gemini-flash-latest`, set in `WRITER_MODEL` in `index.html`.
- Embedding — `models/gemini-embedding-001`, set in `EMBED_MODEL` in `build_dataset.py`.

Google retires model versions, and a pinned name starts returning 404 the day it
happens. That is exactly how `gemini-2.0-flash` and `text-embedding-004` stopped
working here mid-build.

---

## Using it

1. Open the app in Safari.
2. Pick a mood, an author, a book.
3. Tap **Short** or **Long**. The first time, paste a Gemini key when asked.

On an iPhone, Share → **Add to Home Screen** and it opens like an app.

Without a key the catalogue still works completely — every mood, every author,
every book, and the recommendations. Only the written summaries and
out-of-catalogue lookups need one.

---

## Building the dataset

The app ships with about 120 books built in, so it works with nothing else
present. To replace them with a few thousand from the full dump:

```bash
pip install chromadb pyarrow fsspec numpy requests aiohttp
```

```bash
cp .env.example .env
```

Edit `.env` so it reads `GOOGLE_API_KEY=your-real-key`, then:

```bash
python build_dataset.py
```

That writes `books.json` and `vectors.json` next to `index.html`, and the app
picks them up automatically. Check `git status` afterwards to confirm `.env` is
not listed.

**No 2 GB download.** By default the script streams a 3.9-million-row Goodreads
dataset off Hugging Face over HTTPS. Parquet is columnar, so only the seven
columns it reads cross the network, and each batch is discarded after scoring.
Streaming runs at roughly 5,000 rows a second — a full pass is about fifteen
minutes.

| Flag | Does |
|---|---|
| `--scan-limit 100000` | Stop early. Good for a first run. |
| `--source local` | Read the UCSD `.gz` dump instead, if you have it. |
| `--shortlist 60000` | How many top-scoring books get embedded. |
| `--export 3000` | How many end up in `books.json`. |
| `--max-per-author 6` | Stops one prolific author crowding out the rest. |
| `--vectors-only` | Just rebuild `vectors.json` from an existing `books.json`. |
| `--no-vectors` | Skip the Gemini stage entirely. |

Costs of a full run: about fifteen minutes streaming, twenty minutes of CPU to
embed a 60,000-book shortlist locally, and around sixty Gemini calls for the
browser vectors — comfortably inside the free tier.

### The filters, and why each exists

Most of the work in this script is throwing things away, and every filter is
there because of something that actually turned up in the output:

- **Enough signal** — a real blurb, a rating, at least 25 votes.
- **English** — the dataset has no language column and is thick with translated
  editions, so blurbs are scored on English function-word density. Real English
  prose runs about a fifth; other languages score near zero.
- **Actually a novel** — must claim a fiction label, must not claim nonfiction.
  Without this the ranking fills with Bibles, cookbooks and art monographs, which
  are popular and well rated but have no plot.
- **One story** — no omnibuses, boxed sets, samplers, split volumes, study guides
  or tie-in merchandise. Some only confess in the blurb ("books 1–9 in one
  volume") while the title looks innocent, so descriptions are checked too.
- **Author coverage** — mood queries strand authors on one or two books, and an
  author with fewer than three is a dead end in the app. Any author close to
  qualifying has their remaining books tagged with whichever mood they sit
  nearest. On a trial run this took the export from 35 books by 11 authors to 331
  by 85.

The merchandise filter is deliberately narrow. A looser `guide to` rule would
throw out *The Hitchhiker's Guide to the Galaxy*, which is very much a novel.

A long tail of edition oddities survives all this. The filters are grouped near
the top of the script and are easy to extend.

---

## Editing it

Everything is in `index.html`:

- **The books** — the `BOOKS` list at the top of the `<script>` block, used when
  `books.json` is absent. Give every author at least three books.
- **The moods** — the `GENRES` list below it, and `MOODS` in `build_dataset.py`.
  The tags must match.
- **The prompts** — `buildPrompt()` for catalogue books, `buildAskPrompt()` for
  ones we don't have. The spoiler rules live here.
- **The retrieval** — `retrieve()`, which decides what grounding the model gets.
- **The model** — `WRITER_MODEL`, one line.
- **The colours** — five values in `:root` at the top of the `<style>` block.

## Size

`books.json` is about 2.8 MB and `vectors.json` about 950 KB, both downloaded on
first load and then cached by the browser. If that is too much over cellular,
rebuild with a smaller `--export`; 1,500 books roughly halves it.
