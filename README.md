# Brevis

Worth your week, or worth skipping? Name a novel you loved and Brevis finds five
that are genuinely alike, or start from a mood and work down to a book. Every
book comes with what it's about and what readers shelved it as.

One HTML file. No server, no database, no build step, no packages, no accounts,
no tracking, and **no AI calls at runtime**. It works when you double-click it
and it works on a static host.

**Live:** [kaivalya25.github.io/Brevis](https://kaivalya25.github.io/Brevis/)

---

## How it works

Three ideas, and the interesting part is that all three survive having no backend.
For the short version of what kind of project this is, see *Is this RAG?* below.

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

#### Famous novels, guaranteed

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

Famous novels by authors with fewer than three books stay in the catalogue
(*To Kill a Mockingbird*), but the app leaves those authors off the mood → author
ladder so it never dead-ends. They're found through **More like this** and search.

**Results.** The full build found **752 of the 1,500** famous novels with a usable
English edition, and all 752 shipped. The catalogue is now **2,997 books by 768
authors**, 495 of whom have three or more books and appear on the mood ladder.
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
after each build. Searching for one still turns up something: books whose
blurbs mention it, which is often the "for fans of *The Hunger Games*" crowd.

### 2. Semantic retrieval

The shortlisted blurbs are embedded into a local Chroma vector database. Each of
the eleven moods is a sentence — *"a quiet, sad, reflective novel about loss,
memory, grief and regret"* — and querying the index with it is what sorts books
into moods. The same index also gives every book a list of nearest neighbours,
kept as a fallback for when the stronger Gemini vectors below aren't loaded.

None of that runs on your phone. A vector database at runtime would need a
server, so retrieval happens once offline and only the result ships.

`vectors.json` holds one Gemini embedding per book, made at build time: 128
numbers quantised to a byte each, which is why the whole index is under a
megabyte. Everything that needs "books like this one" compares those vectors in
the browser with a dot product: **More like this** and **If you liked this**.
That's arithmetic on data already in the page, so it needs no API call. It's also much better than the build's local model: for *The Way of
Kings* the local model suggested *A Mother's Shame*; the Gemini vectors suggest
*A Memory of Light*, *A Game of Thrones* and *The Dragon Reborn*.

**Hubness correction.** Plain cosine similarity has a known flaw: a few vectors
sit close to everything and show up in almost every neighbour list. Here
*Atonement* appeared in **91** different books' top five. The standard fix is
**CSLS** (cross-domain similarity local scaling). The build stores a "hub"
score per book, its mean similarity to its 25 nearest neighbours, and the app
ranks by `2 × cosine − hub`, which docks books for being close to everything:

| | Plain cosine | With CSLS |
|---|---|---|
| Most times any one book is recommended | 91 (*Atonement*) | 29 |
| Share of the catalogue that ever gets recommended | 74% | 93% |

Measured over every book's top five, offline in numpy and again in the browser,
with identical results. A lookup takes under a millisecond.

### More like this

The first thing on the mood screen. Type a book you loved, pick it from the
matches, and get its five nearest neighbours.

- **Matching** ranks exact titles first, then titles starting with what you
  typed, then any word in the title, then authors. Accents and punctuation are
  ignored, and `way kings` finds *The Way of Kings*.
- **Series names, and books we don't have.** People type *mistborn*, but the
  stored title is *The Final Empire*. If nothing matches a title or author, the
  blurbs are searched for the whole phrase, and the results are labelled
  honestly: "No title by that name. These books mention *mistborn*." The same
  search turns up books pitched at fans of one the catalogue lacks, such as *The
  Hunger Games*. It's only a fallback, because a common word like *love* appears
  in 839 blurbs.
- **Same-author books are left out.** Recommending more Sanderson to someone who
  loved Sanderson tells them nothing; there's a link to the book itself instead.
- **Free and instant.** No key, no API call.

### 3. The book page

Opening a book shows its publisher's blurb from the dataset, the moods it was
tagged with, what readers shelved it as, and four books like it. Nothing is
fetched and nothing is generated, so it's instant and works offline.

---

## Is this RAG?

**No.** RAG means *Retrieval-Augmented Generation*: retrieve relevant material,
then have a language model write an answer from it. Earlier versions of Brevis
did exactly that, retrieving a book's blurb and neighbours and having Gemini
write spoiler-free summaries from them. That generation step has been removed,
along with every other runtime AI call.

What's left is a **retrieval and recommendation system**:

| Part | Technique | When it runs |
|---|---|---|
| Ranking | Bayesian weighted rating, top-K heap over 3.9M rows | Build time |
| Coverage | Wikidata must-include list, fuzzy title matching | Build time |
| Moods | Vector queries against a Chroma index | Build time |
| Embeddings | Local MiniLM (shortlist), Gemini `gemini-embedding-001` (export) | Build time, once |
| Similar books | Dot-product nearest neighbours with CSLS hubness correction | In the browser, no API |
| Search | Fuzzy title, author and series-name matching | In the browser, no API |

Machine learning is still in it, as embedding models that turn blurbs into
vectors. But they run once, on a laptop, when the dataset is built. The app
itself never calls a model.

---

## Using it

1. Open the app in Safari and tap **Begin**.
2. Type a book you loved to get five like it, or pick a mood, then an author,
   then a book.
3. Each book shows its blurb, moods, shelves, and books like it.

On an iPhone, Share → **Add to Home Screen** and it opens like an app.

No key, no account, no sign-up. Nothing a visitor does sends anything anywhere.

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

Edit `.env` so it reads `GOOGLE_API_KEY=your-real-key`. The build is the only
thing that uses a key, for the embeddings in `vectors.json`; the app never does.
Get a free one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Then:

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
| `--no-vectors` | Skip the Gemini stage entirely. More like this then falls back to the weaker local-model neighbours. |

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
- **Author coverage** — mood queries strand authors on one or two books, and an
  author with fewer than three is a dead end in the app. Any author close to
  qualifying has their remaining books tagged with whichever mood they sit
  nearest. On a trial run this took the export from 35 books by 11 authors to 331
  by 85.
- **Famous novels** — the exception to the rules above. See *Famous novels,
  guaranteed*.

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
- **The book page** — `renderAbout()`.
- **Similar books** — `similarTo()` and `nearestBooks()`, including the CSLS
  correction.
- **Search** — `matchTitles()` and `matchBlurbs()`.
- **The must-include list** — `famous_novels.json`. It's plain JSON and safe to
  edit by hand: add a novel you want guaranteed, then rebuild.
- **The colours** — five values in `:root` at the top of the `<style>` block.

## Data sources

- **Goodreads books** — the [BrightData/Goodreads-Books](https://huggingface.co/datasets/BrightData/Goodreads-Books)
  dataset on Hugging Face, streamed, never stored.
- **Famous novels** — [Wikidata](https://www.wikidata.org/), released under
  [CC0](https://creativecommons.org/publicdomain/zero/1.0/). `famous_novels.json`
  is a cached query result and carries the same dedication.
- **Embeddings** — Google Gemini, at build time only.

## Size

`books.json` is about 2.8 MB and `vectors.json` about 1 MB, both downloaded on
first load and then cached by the browser. If that is too much over cellular,
rebuild with a smaller `--export`; 1,500 books roughly halves it.
