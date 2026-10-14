# Job Search Resume Pipeline

A LangGraph pipeline that tailors a resume and cover letter to a job posting and checks every claim against your real experience. About 9 cents per application.

## Why it exists

Tailoring a resume for each job takes time, and AI tools tend to invent things: a metric you never hit, a tool you never used. This project does two things about that.

1. It decides whether a posting is worth applying to before spending money on it.
2. It only lets the writer use facts from a file you wrote yourself. Python checks then reject any bullet that adds a number or a technology that is not in that file.

I built it for my own job search, then cleaned it up so someone else can use it with their own data. A few prompts still describe my profile (see "Using your own data").

## How it works

Every posting goes through the same checks, cheapest first:

1. **Red-flag regex checks.** Free. They reject postings that say things like "no sponsorship" or "security clearance required".
2. **Cache lookup.** Free. Each posting is hashed by its text, so the same posting is not scored twice.
3. **Fit score.** One cheap LLM call. It scores the posting against your profile.
4. **Tailoring.** Only for postings that pass. A 6-node LangGraph pipeline runs in this order: `should_apply`, `triage`, `strategist`, `writer`, `editor`, `finalizer`.
5. **Claim check.** Python validators compare every generated bullet to your master data and revert anything invented or any number that was dropped.
6. **Output.** The resume and cover letter are rendered with Jinja2 and LaTeX into PDFs.

**Which models do what**

- **Claude Haiku** handles scoring, keyword extraction, planning (the strategist step) and editing.
- **Claude Sonnet** writes the final resume and cover letter text.
- **Groq and a local Ollama model are optional alternatives.** The cheap tier uses the first provider that is available, in this order: Haiku (needs `ANTHROPIC_API_KEY`), Groq (needs `GROQ_API_KEY`), then Ollama (must be running). If the chosen provider hits a daily quota, a missing model or repeated tool-call failures during a run, the pipeline moves to the next one.
- **`LLM_CHEAP_PROVIDER` forces one.** Set it to `anthropic`, `groq` or `ollama`.
- `QUALITY_MODEL` changes the Sonnet model. `core/llm.py` explains why Haiku is first: Groq's free model gave inconsistent scores in testing.

Three ways to run the same engine:

| Entry point | What it is |
|---|---|
| `server.py` | Web UI (FastAPI backend, plain JavaScript frontend) |
| `apply.py` | Command line |
| `run.py` | Runs the pipeline from environment variables, so it can be called from GitHub Actions. This repo only ships the test workflow. |

The scraper collects postings from LinkedIn and Indeed (through python-jobspy), Dice, Google Jobs (through SerpApi) and company career boards (Greenhouse, Lever, Ashby, Workday), then writes them to a Google Sheet.

Folders:

| Folder | Contents |
|---|---|
| `core/` | Model selection, cost tracking, red-flag regexes, fit scoring, posting cache |
| `engine/` | The LangGraph pipeline, claim validators, PDF generation |
| `modes/` | Manual run, scraper run, form autofill |
| `scraper/` | Job scraping, filtering, Google Sheets output |
| `cover_letter/` | Regenerate a cover letter from an earlier run |
| `frontend/` | Web UI, no build step |
| `templates/` | LaTeX templates |
| `tests/` | pytest suite |
| `docs/` | Deployment guide |

## How to run it

You need Python and a LaTeX install that provides `pdflatex`. The CI workflow uses Python 3.13 and the Dockerfile uses 3.12.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in the values
cp Ankush_Master_Data.example.json Ankush_Master_Data.json   # then replace with your own data
```

`ANTHROPIC_API_KEY` is required. Everything else is optional or only needed for one mode (SerpApi, Google Sheets and Telegram for the scraper, `APP_PASSWORD` for the web UI).

**Web UI**

```bash
./venv/bin/uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000` and unlock it with `APP_PASSWORD`. On the **Add JD** tab, enter a company, a title and the job description, then click **Add to list**. The posting is scored, and you can generate a tailored resume (and cover letter) for it from the **Applications** tab. The **Scraper** tab runs the job scraper.

**Command line**

```bash
./venv/bin/python apply.py score --jd-file jd.txt        # fit verdict only
./venv/bin/python apply.py manual --company "Stripe" --title "SWE" --jd-file jd.txt --cover-letter
./venv/bin/python apply.py scrape [--daemon]             # one scrape cycle, or run on a schedule
./venv/bin/python apply.py autofill --url <application url> --folder <run output folder>
./venv/bin/python apply.py cache-stats                   # cached postings, pass and reject counts
```

`autofill` fills in an application form with Playwright and stops before submitting. It does not answer visa, EEO or salary questions.

**Resume with no job description, no LLM, no cost**

```bash
./venv/bin/python engine/generate_generic.py
```

**Tests**

```bash
./venv/bin/pytest tests/ -v
```

The tests cover deterministic logic only (regex checks, the cache, PDF text cleanup, claim checks). They make no paid calls. `conftest.py` sets a placeholder API key so the suite cannot spend money.

### Using your own data

`Ankush_Master_Data.example.json` is a fictional person ("Alex Rivera") in the format the engine reads: contact info, summary, skills, employers with bullets, projects, education. Copy it to `Ankush_Master_Data.json` (which is gitignored) and replace every value with your own real experience.

It also has a `candidate_context` section, which the code reads instead of hardcoded text:

- `scoring_context`: a list of facts for the scoring prompt, such as years of experience, work authorization and target level.
- `visa_line`: one line for the profile summary.
- `title_corrections` (optional): job titles the model tends to get wrong for an employer, so the checks can catch them.

Some things were written for my profile and you should review them:

- `engine/graph.py`: prompt examples reflect the author's own background and can be edited.
- `core/red_flags.py`: rejects "no sponsorship" and clearance postings, which suits a candidate who needs sponsorship. Edit it if that is not you.
- The Scraper tab in the web UI: roles, location, sources, watched companies, daily cap and minimum score.

### Optional: LinkedIn cookie

LinkedIn scraping works without a login but returns less. If you set `LINKEDIN_COOKIE` (your own `li_at` session value), results improve. Using a logged-in session for automated access is against LinkedIn's terms and can get your account restricted. It is off unless you set it. Never commit the cookie. `.env` is gitignored.

## Results

Cost is measured from the token counts the API bills, by `core/usage.py`. Each tailored run writes a `usage_report.json` into its output folder, and scraper runs record their measured cost in the run history.

| Step | Cost |
|---|---|
| Tailored application, resume only | 8.7 cents |
| Tailored application, resume plus cover letter | 8.9 cents |
| Scoring one posting | under 1 cent (0.46 cents measured) |

The two tailoring figures are from two real runs on the same posting, so treat "about 9 cents" as an estimate. Prompt caching did not apply to tailoring in those runs. The scoring figure comes from a scrape that scored 80 postings and was billed $0.37 in total. Postings rejected by the regex checks or found in the cache cost nothing.

## Tech stack

Python, LangGraph, LangChain, Claude (Haiku and Sonnet), Groq and Ollama (optional), FastAPI, Playwright, python-jobspy, SerpApi, Google Sheets (gspread), Jinja2, LaTeX, pytest, Docker, Fly.io. Plain HTML, CSS and JavaScript for the web UI.

Deployment notes are in `docs/DEPLOYMENT.md`.

This is a personal project and is not affiliated with any job board or employer. MIT license.
