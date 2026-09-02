# Brevis — Build Spec

Hand this file to Claude Code. Start a session in the project folder and say:

> Read PROJECT.md and build it.

\---

## What we're building

A single-page book recommendation web app called **Shelf**, run from an iPhone home screen. Flow:

**Welcome → pick a genre → pick an author → pick a book → read a 1000–1500 word full-plot summary.**

Every part must cost $0. No backend, no database, no build step, no npm packages, no framework, no analytics, no paid tier anywhere. The only network call at runtime is an optional one to Google Gemini, using an API key the user pastes into the app themselves.

## Hard constraints

* **One file**: `index.html` with CSS and JS embedded. Vanilla only — no React, Vue, Tailwind, jQuery, or CDN imports.
* **Runs from `file://` and from a static host** with no server logic.
* **Comments throughout**, written so a non-expert can find and edit the book list, the colours, and the Gemini prompt.
* **No API key in the code.** The user types it in at runtime; store it in `localStorage` behind a try/catch wrapper so private browsing doesn't throw.
* Mobile-first. Tap targets ≥ 44px, `viewport-fit=cover`, safe-area padding, no hover-dependent UI.

## Files to produce

|File|Purpose|
|-|-|
|`index.html`|The entire app|
|`build\_dataset.py`|Optional, offline only. Curates a bigger `books.json` from the Goodreads dump|
|`.gitignore`|Ignore `\*.gz`, `books.json`, `.env`, `\_\_pycache\_\_/`|
|`README.md`|Short plain-English readme for the repo|

\---

## `index.html` specification

### Screens

Six `<section class="screen">` blocks, one visible at a time via a `.on` class toggled by a `go(id)` function. No routing, no history API.

1. **Welcome** — app name, one-line description, Start button, and a quiet link to API key settings.
2. **Genre** — a 2-column grid of buttons, one per genre. Single select; tapping advances immediately.
3. **Author** — the top 5 authors in the chosen genre as buttons, plus a search input below them.
4. **Books** — 3 to 5 of that author's books as buttons, each showing rating, year and a one-line hook.
5. **Summary** — the long AI summary, with "Other books" and "Start over" buttons.
6. **Settings** — password-type input for the Gemini key, Save and Remove buttons, and a line stating the current state.

Every screen after the first has a back link.

### Data

Inline `const BOOKS = \[...]` array. Aim for **\~120 books across \~50 authors**, and make sure **most authors have 3–4 titles** — otherwise the author→books step dead-ends. Use real, well-known books so Gemini can summarise them accurately.

Each entry:

```js
{ t:"Title", a:"Author", r:4.3, y:2015, g:\["fantasy","dark"], d:"One-line hook." }
```

Genres (tag, label): cozy/Cozy, thrilling/Thrilling, romantic/Romantic, sci-fi/Sci-fi, fantasy/Fantasy, melancholic/Melancholic, funny/Funny, mind-bending/Mind-bending, historical/Historical, dark/Dark, classic/Classic.

Aim for at least 8 books per genre so the top-5 author ranking is meaningful.

### Top 5 authors logic

Filter books to the chosen genre, group by author, average their ratings **within that genre only**, then sort. Add a small consistency nudge so a writer with several strong books in the genre outranks a single lucky 4.5:

```js
score = avgRating + Math.min(bookCount, 3) \* 0.05
```

Take the top 5. Each button shows the author name, the average to 2dp, and how many of their books qualified.

### Author search

Input below the top five. Searches **every author in the dataset**, not just the current genre, so a user who entered through Cozy can still reach Cormac McCarthy. Fires on `input`, ignores queries under 2 characters, shows up to 6 matches with each author's overall average and book count.

### Book list

That author's books in the chosen genre first, sorted by rating descending; if fewer than 3, top up with the rest of their work so there's never a dead end. Cap at 5.

### The summary

On tapping a book, POST to:

```
https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=KEY
```

Body: `{ contents:\[{parts:\[{text: prompt}]}], generationConfig:{ temperature:0.4, maxOutputTokens:3000 } }`
Read the result from `data.candidates\[0].content.parts`.

The prompt must ask for 1000–1500 words covering: the entire plot including the ending, all major twists and where each one lands, main characters and their motivations and arcs, setting, themes and tone. It must specify plain prose in past tense, no headings, no bullets, no markdown, paragraphs separated by blank lines, **no quoting any text from the book**, and an instruction to say so plainly rather than inventing events if the model isn't confident about the real plot.

Split the response on blank lines into `<p>` elements. Escape all interpolated text — never write raw model output or book fields into `innerHTML` unescaped. Show a "Writing the summary…" state while it's in flight. Cache results in a plain object keyed by title so re-opening a book doesn't spend quota again.

Fallbacks: no key saved → show the one-line hook plus a note pointing at settings. Request fails → show the hook plus a plain error explaining what to check. Never a blank screen, never a raw stack trace.

### Design

&#x20;The website, the name of the app should be Brevis. All the aesthetics, font and animations should exactly be like Squarespace landing page.

iPhone meta tags

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Shelf">
<meta name="theme-color" content="#ffffff">
```

\---

## `build\_dataset.py` specification

Offline only — runs on the laptop, never on the phone. ChromaDB is a build-time tool here; using it at runtime would need a server and break the $0 rule.

Reads `goodreads\_books.json.gz` and `goodreads\_book\_authors.json.gz` from the UCSD dataset (https://cseweb.ucsd.edu/\~jmcauley/datasets/goodreads.html). Streams the gzip line by line rather than loading 2.3M records into memory. Keeps books with ratings\_count ≥ 50,000, average\_rating ≥ 3.5, an English language code and a description over 200 characters. Joins author names by `author\_id`.

Embeds descriptions with `models/text-embedding-004` in batches of 100 with a short sleep between calls, loads them into a local Chroma collection, then queries once per mood using a short descriptive phrase for that mood, taking the top 30 per mood. Dedupes by title, letting one book carry several mood tags. Writes `books.json` in exactly the `{t,a,r,y,g,d}` shape above, using the first sentence of the blurb as the hook.

Requires `pip install chromadb google-generativeai` and `GOOGLE\_API\_KEY` in the environment.

\---

## After the code is written

Claude Code should do these itself, in order.

**1. Preview.** Serve the folder on port 8000 (`python -m http.server 8000`) and print the URL. A local server is required — `fetch()` for `books.json` is blocked on `file://`.

**2. Sanity checks.** Confirm: every screen reaches the next and back; picking each of the 11 genres produces 5 author buttons; every author reachable through the flow has at least 3 books; searching "aus" returns Jane Austen; no `books.json` fetch exists yet if the data is inline; no console errors on load.

**3. Ship it.**

```bash
git init
git add .
git commit -m "Brevis: book summarizer"
gh repo create shelf --public --source=. --push
gh api -X POST repos/:owner/brevis/pages -f "source\[branch]=main" -f "source\[path]=/"
```

If `gh auth status` fails, stop and tell me to run `gh auth login` first. Then print the live URL: `https://<username>.github.io/brevis/`.

Do not commit an API key. If one is ever hardcoded during testing, remove it before the first commit — a key in a public repo gets scraped within hours.

**4. Report back** with the live URL and a one-line reminder to open it in Safari and use Share → Add to Home Screen.

## Explicitly out of scope

Backend servers, databases, user accounts, npm or any build tooling, service workers, offline caching, tracking or analytics, cookie banners, paid APIs or hosting tiers, ratings/reviews features, reading lists, and any placeholder feature not described above. Keep the code minimal — no unused CSS resets, no dead functions.

