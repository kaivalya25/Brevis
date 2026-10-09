# Brevis

**Loved a book? Find five more.** Brevis is a retrieval-augmented ("RAG")
book recommender. Type a novel you loved; it retrieves the closest books in a
catalogue of about three thousand by what their blurbs are about, then Gemini
picks the five you're most likely to love and says what connects each one,
using only those books' own descriptions.

One HTML file. No server, no database, no build step in the app, no packages, no
accounts, no tracking. It works when you double-click it and it works on a
static host.

**Live:** [kaivalya25.github.io/Brevis](https://kaivalya25.github.io/Brevis/)

---

## How a recommendation is made

```
 the book you loved
        │
        ▼
 1. RETRIEVE   its 12 nearest books in vector space          in the browser, no AI call
        │      (Gemini embeddings + CSLS hubness correction)
        ▼
 2. AUGMENT    a prompt with the loved book and those 12:
        │      blurbs, moods, what readers shelved them as
        ▼
 3. GENERATE   Gemini picks the best 5 from the 12,          one API call
        │      one sentence each on what connects them
        ▼
 4. CHECK      every pick must be one of the 12, used once; anything else is dropped
        │      short of 5? filled from retrieval order, with no invented reason
        ▼
 five books, each with a reason
```

### 1. Retrieval

Every book in the catalogue has a vector: a Gemini embedding of its blurb, made
once when the dataset is built and shipped in `vectors.json`. That's 128 numbers
per book, quantised to a byte each, so the whole index is under a megabyte.
Finding the books most like yours is a dot product in the browser, with no API
call and no server.

**Hubness correction.** Plain cosine similarity has a known flaw: a few vectors
sit close to everything and show up in almost every neighbour list. Here
*Atonement* appeared in **91** different books' top five. The standard fix is
**CSLS** (cross-domain similarity local scaling). The build stores a "hub" score
per book, its mean similarity to its 25 nearest neighbours, and retrieval ranks
by `2 × cosine − hub`:

| | Plain cosine | With CSLS |
|---|---|---|
| Most times any one book is recommended | 91 (*Atonement*) | 29 |
| Share of the catalogue that ever gets recommended | 74% | 93% |

Books by the same author are left out. Recommending more Sanderson to someone
who loved Sanderson tells them nothing.

### 2–3. Augmented generation

The 12 retrieved candidates go into the prompt beside the book you loved, each
with its blurb, moods and shelves, and numbered. Gemini is asked to:

- pick the 5 this reader is most likely to love, best first;
- give one sentence (30 words at most) each on what specifically connects it to
  the book they loved: a shared theme, setting, tone, structure or kind of
  character;
- use **only what the blurbs and labels say**, not whatever else it remembers
  about those books;
- reveal nothing about a plot beyond what its blurb says;
- reply as JSON (`responseMimeType: application/json`): `[{"id": 3, "why": "…"}]`.

Retrieval decides *which* books are in the running; generation decides the order
and explains it. That's the division of labour RAG is for: the model works from
the material it's handed rather than from memory.

### 4. Checking the answer

The model answers with candidate **numbers**, never titles, and the app checks
every one:

- **A number that isn't a candidate is dropped.** The app can never recommend a
  book that isn't in the catalogue, however the model misbehaves.
- **A candidate picked twice counts once.**
- **An unreadable answer** (not JSON, cut short) falls back to retrieval order.
- **Fewer than five valid picks** are topped up from retrieval order, shown
  without a reason rather than with an invented one.

In testing, a simulated Gemini that returned a made-up candidate number and a
duplicate had both thrown out, with the gaps filled from retrieval.

### Graceful degradation

Generation makes the results better, but it's never required:

| Situation | What you get |
|---|---|
| A Gemini key is available | Five books chosen and explained by Gemini |
| No key | The five nearest books, from retrieval alone, plus a box to add a key |
| Gemini busy (503/429) | Retried twice, after 1s and 2s, then a backup model (`gemini-flash-lite-latest`); then the retrieved books with a **Try again** button |
| Key rejected | The retrieved books, plus a box to paste a different key |
| Opened before | Instant: answers are cached, in memory and in the browser |

A slow answer that arrives after you've moved to another book is dropped, not
shown on the wrong page.

### Books the catalogue doesn't have

Searching for a book Brevis lacks offers **Find books like "…"**. There's no
stored vector to start from, and embedding a bare title works badly: *Gone Girl*
landed next to a book called *Gone*, because a title is a few words with no
meaning attached. So Brevis uses **HyDE** (hypothetical document embeddings):

1. Gemini writes a two-sentence blurb for the book, which is shown on the page so
   a misidentified book is obvious. A title it doesn't recognise is refused,
   not guessed at.
2. That blurb is embedded with the same model as the catalogue and used to
   retrieve the 12 nearest books.
3. Steps 2–4 above run as usual, with the written blurb standing in for the
   loved book's.

### Finding the book you loved

Plain text matching, no AI: exact titles first, then titles starting with what
you typed, then any word in the title, then authors. Accents and punctuation are
ignored, and `way kings` finds *The Way of Kings*. If nothing matches a title
or author, blurbs are searched for the whole phrase, which finds series names
(*mistborn* finds *The Final Empire*). Those results are labelled "These books
mention *mistborn*", never presented as the book itself.

---

## The catalogue

`build_dataset.py` builds the catalogue offline, on a laptop. It reads a
Goodreads dataset of 3.9 million books and ranks every one with a weighted
rating, the standard fix for a book with nine five-star reviews outranking
every classic ever written:

```
score = (v / (v + m)) · R  +  (m / (v + m)) · C
```

`R` is the book's own average, `v` how many people rated it, `C` the average
across the whole dataset, and `m` how many votes it takes before a book is
trusted to speak for itself. A heap keeps only the best few thousand as the
millions stream past, so memory stays flat. The shortlist is embedded into a
local Chroma vector database, which sorts books into eleven moods (each mood is
a sentence the index is queried with) and gives every book a fallback list of
neighbours. Gemini then embeds the exported books for `vectors.json`.

### Famous novels, guaranteed

Ratings alone missed the books people type first. The weighted score and the
"authors with three or more books" rule are both sensible, and between them they
had quietly dropped *Gone Girl*, *Pride and Prejudice*, *The Hunger Games*,
*Outlander* and *The Martian*.

So there's a second, independent signal of fame: **Wikidata**. Every novel there
records how many language editions of Wikipedia have an article about it. A novel
written up in forty languages is famous in a way a star rating can't measure.
The 1,500 most widely covered novels become a must-include list:

- **Fetched once and committed** as `famous_novels.json`, so builds are
  repeatable and don't depend on Wikidata being up. `--refresh-famous` re-fetches.
- **Matched while streaming.** Each of the 3.9M records is checked against the
  list. Of all a famous novel's editions, the one with the most ratings is kept,
  and it skips the vote threshold and the competition for a shortlist place. It
  still has to be English, have a real blurb, and be one story rather than an
  omnibus or study guide.
- **Guaranteed a place** in the export, ahead of everything else.

Wikidata marks a novel with *form of creative work* (P7937) = *novel* (Q8261).
The obvious alternatives, *instance of novel* and *genre: novel*, return nothing.
P7937 found all thirteen books in a test set.

Matching titles across two sources is where the care went. The two disagree in
predictable ways: *Mistborn: The Final Empire* against *The Final Empire*, *The
Hunger Games (The Hunger Games, #1)*, *Nineteen Eighty-Four* against *1984*. So
each title is reduced to plain variants, Wikidata's English aliases are matched
too, and the author's surname has to agree. A trial run then showed what that let
through, and each problem got its own fix:

| Matched wrongly | Why | Fix |
|---|---|---|
| *The Hobbit: The Desolation of Smaug* (a film guide) | Part of the title matched | A whole-title match always beats a part-title match, however many ratings the part-match has |
| *Dune: Red Plague* by **Brian** Herbert | Only surnames were compared | First initials must agree too |
| *Fahrenheit 451: Novel-Ties Study Guide*, *Remembrance of Things Past: Vol 2* | Must-include books skipped the "one story" filter | They no longer skip it, and the filter now catches volumes |
| *Harper Lee's To Kill a Mockingbird* by Harold Bloom (a study guide) | Its title contains the famous title | A famous author's name + 's + their famous title, by someone else, is a companion and is dropped. *Ender's Game* and *The Handmaid's Tale* are unaffected |
| *Frankenstein: The 1818 Text* (1.6M ratings) not matched | Wikidata writes *Frankenstein; or, The Modern Prometheus*, with a semicolon | Titles are split at semicolons as well as colons. This fix landed after the current data was built, so it takes effect on the next build |

Famous novels by authors with fewer than three books (*To Kill a Mockingbird*) are
kept too, whatever the build's other rules say about author depth.

**Results.** The full build found **752 of the 1,500** famous novels with a usable
English edition, and all 752 shipped. The catalogue is now **2,997 books by 768
authors**.
Newly included: *Pride and Prejudice*, *Outlander*, *To Kill a Mockingbird*,
*1984*, *The Great Gatsby* and *Dune*, among others.

**What it can't fix.** The other 748 aren't in the dataset in any usable form,
and that includes some of the biggest bestsellers. A full-dataset search for
them found only what surrounds the book, never the book itself:

| Novel | What the dataset has instead |
|---|---|
| *The Hunger Games* | A trilogy box set, a movie companion, a "Tribute Guide" |
| *Gone Girl* | A "Review" knockoff, two parodies, a "Sidekick" summary |
| *Harry Potter and the Philosopher's Stone* | A pop-up book, a piece of fan fiction |
| *The Name of the Wind*, *Normal People*, *The Martian* | Nothing usable |

The filters rightly reject all of those, and no matching logic can find a book
the scrape never captured. Closing this gap needs a second source of blurbs for
the missing novels, such as Open Library. They're listed in `famous_missing.txt`
after each build. Searching for one still works: **Find books like…** has
Gemini describe it and searches by meaning (see *Books the catalogue doesn't
have*), and the plain search shows books whose blurbs mention it, which is often
the "for fans of *The Hunger Games*" crowd.

---

## The API key

The generation step needs a Gemini key; retrieval doesn't.

- **A visitor's own key.** Asked for inline the first time it's needed, then
  kept in their browser. It's sent only to Google.
- **A shared key, optional.** `config.js` sets `window.BREVIS_SHARED_KEY` for
  visitors who have no key of their own. `write_config.py` writes it from a key
  in your `.env`, so it's never typed or pasted:

  ```bash
  python write_config.py
  ```

  Then commit and push `config.js`. A visitor's own key always takes priority.

**A shared key is public.** Brevis is a static page with no server, so anything
the page can read, a visitor can read, including `config.js`. Anyone could copy
the key and spend its quota. Use a key from a **free-tier Google project with no
billing**, so the worst case is a used-up quota rather than a bill. If
explanations stop appearing, replace the key in `.env`, re-run
`write_config.py`, and push. GitHub may block the push, or report the key to
Google, which can disable it. The fully safe alternative is a small proxy, such
as a free Cloudflare Worker, that keeps the key on a server.

Without `config.js`, the browser console shows one harmless "not found" error
for it, and visitors are asked for their own key.

### Models

All three are aliases rather than pinned versions. Google retires model
versions, and a pinned name starts returning 404 the day it happens, which is
how `gemini-2.0-flash` and `text-embedding-004` both stopped working mid-build.

| Job | Model | Set in |
|---|---|---|
| Choosing and explaining recommendations; describing missing books | `models/gemini-flash-latest` | `WRITER_MODEL`, `index.html` |
| Backup when that one is overloaded | `models/gemini-flash-lite-latest` | `FALLBACK_MODEL`, `index.html` |
| Embeddings, for the catalogue and for missing-book queries | `models/gemini-embedding-001` | `EMBED_MODEL`, `build_dataset.py` |

---

## Using it

1. Open the app in Safari.
2. Type a book you loved and pick it from the matches.
3. Read the five books like it, and why each one fits. Tap one to see five more
   like *that*. Back retraces your steps.

Not in the catalogue? Tap **Find books like "…"**.

On an iPhone, Share → **Add to Home Screen** and it opens like an app.

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

Edit `.env` so it reads `GOOGLE_API_KEY=your-real-key` (a free key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)). The build uses
it for the embeddings in `vectors.json`. Then:

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
| `--famous 1500` | How many of Wikidata's most widely covered novels must be included. `0` switches it off. |
| `--refresh-famous` | Re-fetch the must-include list instead of using `famous_novels.json`. |
| `--vectors-only` | Just rebuild `vectors.json` from an existing `books.json`. |
| `--no-vectors` | Skip the Gemini stage. Retrieval then falls back to the weaker local-model neighbours, and books outside the catalogue can't be searched. |

Costs of a full run: about fifteen minutes streaming, twenty minutes of CPU to
embed a 60,000-book shortlist locally, one Wikidata query (skipped when
`famous_novels.json` is already there), and around sixty Gemini embedding calls
for `vectors.json`, comfortably inside the free tier.

The embedding model is `models/gemini-embedding-001` (`EMBED_MODEL`), an alias
rather than a pinned version. Google retires model versions, and a pinned name
starts returning 404 the day it happens, which is how `text-embedding-004`
stopped working here mid-build.

The famous novels that couldn't be found are written to `famous_missing.txt`
(not committed), which shows where the dataset's gaps are.

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
- **Author depth** — the export favours authors with at least three books, and
  any author close to qualifying has their remaining books tagged with whichever
  mood they sit nearest. This dates from an earlier mood → author → book
  version of Brevis; it still gives the catalogue depth. On a trial run it took
  the export from 35 books by 11 authors to 331 by 85.
- **Famous novels** — the exception to the rules above. See *Famous novels,
  guaranteed*.

The merchandise filter is deliberately narrow. A looser `guide to` rule would
throw out *The Hitchhiker's Guide to the Galaxy*, which is very much a novel.

A long tail of edition oddities survives all this. The filters are grouped near
the top of the script and are easy to extend.

---

---

## Editing it

Everything is in `index.html`, in numbered sections:

- **Retrieval** — `retrieve()` and `nearestBooks()`, including CSLS. `POOL` sets
  how many candidates generation sees (12); `SHOW` how many are shown (5).
- **The prompt** — `buildRecPrompt()`. The grounding and no-spoiler rules live
  here.
- **Checking the answer** — `parseChoices()`.
- **Missing books** — `describePrompt()` and `openOutside()`.
- **Search** — `matchTitles()` and `matchBlurbs()`.
- **The seed list** — `BOOKS` at the top of the `<script>` block, used when
  `books.json` is absent (for example when opened from a `file://` URL).
- **The must-include list** — `famous_novels.json`, plain JSON and safe to edit
  by hand; add a novel, then rebuild.
- **The colours** — five values in `:root` at the top of the `<style>` block.

## Data sources

- **Goodreads books** — the [BrightData/Goodreads-Books](https://huggingface.co/datasets/BrightData/Goodreads-Books)
  dataset on Hugging Face, streamed, never stored.
- **Famous novels** — [Wikidata](https://www.wikidata.org/), released under
  [CC0](https://creativecommons.org/publicdomain/zero/1.0/). `famous_novels.json`
  is a cached query result and carries the same dedication.
- **Embeddings and recommendations** — Google Gemini.

## Size

`books.json` is about 2.8 MB and `vectors.json` about 1 MB, both downloaded on
first load and then cached by the browser. If that's too much over cellular,
rebuild with a smaller `--export`; 1,500 books roughly halves it.
