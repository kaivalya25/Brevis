# Brevis

Pick a mood, pick an author, pick a book, and read the whole plot — twists,
ending and all — in about ten minutes.

Brevis is one HTML file. No server, no database, no build step, no packages,
no accounts, no tracking. It works when you double-click it and it works on a
static host. The only thing it ever sends over the network is a request to
Google Gemini to write a summary, using a key you paste in yourself.

## Using it

1. Open `index.html`, or the hosted version, in Safari.
2. Tap **API key settings** and paste in a free Gemini key from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey). The key is
   kept in your browser only — Brevis has nowhere else to put it.
3. Tap **Begin**, choose a mood, choose an author, choose a book.

On an iPhone, tap Share → **Add to Home Screen** and it opens like an app.

Without a key everything still works; you just get the one-line hook for each
book instead of the full summary.

## Editing it

Everything is in `index.html`, near the top of the `<script>` block:

- **The books** — the `BOOKS` list. Each line is one book:
  title, author, rating, year, mood tags, and a one-line hook. Give every
  author at least three books, and every mood at least eight.
- **The moods** — the `GENRES` list, right below the books.
- **The Gemini prompt** — inside `buildPrompt()`, further down.
- **The colours** — the five values in `:root` at the top of the `<style>` block.

## build_dataset.py

Optional and offline. If you want a much bigger book list, this script reads
the UCSD Goodreads dump, embeds the blurbs with Gemini, sorts them into moods
with a local Chroma collection, and writes `books.json` in the same shape as
the list above. It runs on a laptop, never on the phone.

```bash
pip install chromadb google-generativeai
export GOOGLE_API_KEY=...
python build_dataset.py
```

## A note on the API key

Never commit a key. `.gitignore` covers `.env`, and there is no key anywhere in
`index.html` — the app asks for one at runtime. A key pushed to a public repo
gets scraped within hours.
