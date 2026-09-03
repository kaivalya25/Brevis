# Brevis

Pick a mood, pick an author, pick a book, and read what actually happens in it —
twists, ending and all — in either three minutes or ten.

Brevis is one HTML file. No server, no database, no build step, no packages, no
accounts, no tracking. It works when you double-click it and it works on a
static host. The only thing it ever sends over the network is a request to
Google Gemini, using a key you paste in yourself.

**Live:** https://kaivalya25.github.io/Brevis/

---

## How it works

Three ideas, and the interesting part is that all three survive having no backend.

### 1. A recommendation system

`build_dataset.py` reads a Goodreads dump of several million books and ranks
every one of them with a weighted rating — the standard fix for the problem
that a book with nine five-star reviews would otherwise outrank every classic
ever written:

```
score = (v / (v + m)) · R  +  (m / (v + m)) · C
```

where `R` is the book's own average, `v` is how many people rated it, `C` is the
average rating across the whole dump, and `m` is how many votes it takes before
a book is trusted to speak for itself. A heap keeps only the best few thousand
as the millions stream past, so memory stays flat.

Inside the app, authors are ranked a second time — averaged within the mood you
picked, with a small nudge for having written several good books in it rather
than one lucky one.

### 2. Retrieval-augmented generation

The shortlisted blurbs are embedded into a local Chroma vector database. Each
of the eleven moods is a sentence — *"a quiet, sad, reflective novel about loss,
memory, grief and regret"* — and querying the index with it is what sorts books
into moods. The same index gives every book its nearest neighbours, which is
where **If you liked this** comes from.

None of that runs on your phone. A vector database at runtime would need a
server, which would cost money, so the retrieval is done once offline and only
the result ships: a few thousand books, each carrying its blurb and the
positions of its neighbours.

At runtime the app still retrieves before it generates. Opening a book gathers
its blurb, rating, moods and nearest neighbours out of the local dataset and
puts them in the prompt as grounding, with an instruction to trust that material
over the model's own memory. That is the *augmented* half.

The search box on the mood screen is genuine vector search in the browser: your
phrase goes to Gemini's embedding model, comes back as 128 numbers, and is
compared against one stored vector per book. Same model, same size, same space,
so a dot product ranks them. The vectors are quantised to a byte each, which is
why the whole index is a few hundred kilobytes rather than several megabytes.

### 3. The summary

Gemini writes it, grounded in the retrieved context, in one of two lengths:

- **Short** — 250–400 words. Premise, arc, ending.
- **Full plot** — 1000–1500 words. Every twist and where it lands, every
  character's motivation, the setting, the themes.

Both are cached per book, so switching back and forth is free. The prompt asks
for plain past-tense prose with no markdown, no quotation from the book, and an
honest admission rather than invention when the model is not sure of the real
plot.

---

## Using it

1. Open the app in Safari.
2. Tap **API key settings** and paste in a free Gemini key from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey). It is kept
   in your browser only — Brevis has nowhere else to put it.
3. Tap **Begin**.

On an iPhone, Share → **Add to Home Screen** and it opens like an app.

Without a key everything still works; you get each book's blurb instead of a
written summary.

---

## Building the dataset

The app ships with about 120 books built in, so it works with nothing else
present. To replace them with a few thousand pulled from the full dump:

```bash
pip install chromadb pyarrow fsspec numpy requests aiohttp google-generativeai
python build_dataset.py
```

That writes `books.json` next to `index.html`, and the app picks it up
automatically.

**No 2 GB download.** By default the script streams a 3.9-million-row Goodreads
dataset off Hugging Face over HTTPS. Parquet is columnar, so only the seven
columns it actually reads cross the network, and each batch is discarded after
scoring. Nothing is written to disk except the final `books.json`.

Useful flags:

| Flag | Does |
|---|---|
| `--scan-limit 100000` | Stop early. Good for a first run. |
| `--source local` | Read the UCSD `.gz` dump instead, if you have it. |
| `--shortlist 60000` | How many top-scoring books get embedded. |
| `--export 3000` | How many end up in `books.json`. |
| `--no-vectors` | Skip the Gemini stage; search falls back to author names. |

A `GOOGLE_API_KEY` is needed only for the last stage, which embeds the exported
books so the browser can search them by meaning. Everything before it runs
offline and free.

Rough costs of a full run: a few hours of CPU for the local embeddings, and
around thirty Gemini API calls — comfortably inside the free tier.

---

## Editing it

Everything is in `index.html`:

- **The books** — the `BOOKS` list at the top of the `<script>` block. Give
  every author at least three books, and every mood at least eight.
- **The moods** — the `GENRES` list below it, and `MOODS` in `build_dataset.py`.
- **The prompt** — `buildPrompt()`.
- **The retrieval** — `retrieve()`, which decides what grounding the model gets.
- **The colours** — five values in `:root` at the top of the `<style>` block.

## A note on the API key

Never commit one. `.gitignore` covers `.env`, and there is no key anywhere in
`index.html` — the app asks for one at runtime. A key pushed to a public repo
gets scraped within hours.
