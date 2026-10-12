# AGENTS.md — ComicCraft Project Rules

This file defines mandatory rules for coding agents working on ComicCraft.
Follow these rules for all implementation, refactoring, and documentation tasks.

## 1. Project

* Name: ComicCraft
* A web-based AI comic generator.
* Backend: FastAPI + Jinja2.
* Language: Python 3.12.
* Development environment: Windows.
* Use the project's `.venv` for all Python execution, installs, and tests.

## 2. Core Architecture

Use this pipeline — do not reorder or skip steps without explicit approval:

```
User Input
→ Validate Input
→ Text provider creates a structured 5-panel comic outline
→ Text provider expands the outline into detailed narration, dialogue, scene details, and image prompts
→ Hosted Hugging Face inference generates panel illustrations
→ Build a canonical Comic object
→ Preview comic
→ Export comic to PDF
```

* The text-generation provider is configurable via `COMIC_TEXT_PROVIDER` (default `"hf"`).
* `"hf"` uses the Hugging Face hosted text provider (`app/hf_text.py`).
* `"gemini"` remains available as an alternative provider (`app/gemini_flash.py`, `app/gemini_pro.py`).

* Keep AI logic separate from HTTP routes.
* Routes must remain thin and must call dedicated services/modules.
* No AI API calls, prompt construction, or image/PDF logic directly in route handlers.

## 3. Canonical Data Model

Use Pydantic models as the single source of truth.

`Comic` must contain:

* `title`
* `prompt`
* `character`
* `setting`
* `tone`
* `art_style`
* `panels`

`Panel` must contain:

* `panel_number`
* `title`
* `scene_description`
* `narration`
* `dialogue`
* `image_prompt`
* `image_path`

Rules:

* The same `Comic` object/data structure must feed both the preview and PDF exporter.
* Do not define parallel/divergent dict shapes for preview vs. export.
* Validate all AI-generated structured output against these models before using it.

## 4. Planned Modules

Keep responsibilities separated:

* `app/main.py` — FastAPI application setup only
* `app/routes.py` — HTTP routes only (thin)
* `app/hf_text.py` — hosted HF outline + detailed story generation
* `app/gemini_flash.py` — outline generation (5-panel structured outline)
* `app/gemini_pro.py` — detailed story generation (narration, dialogue, scene details, image prompts)
* `app/image_generator.py` — hosted image generation
* `app/layout_builder.py` — comic layout preparation
* `app/exporters.py` — PDF export
* `templates/` for UI
* `static/panels/` for generated panel images
* `static/exports/` for generated PDFs

Do not put the entire application into `main.py`.

## 5. AI Providers

* Text/story generation uses a configurable provider (`COMIC_TEXT_PROVIDER`, default `"hf"`).
* `"hf"` uses hosted Hugging Face Inference Providers for text generation.
* `"gemini"` (Gemini Flash + Gemini Pro) remains available as an alternative text provider.
* Hugging Face hosted inference is used for image generation.
* Never assume historical model IDs are still available.
* Model IDs/providers must be configurable through environment variables.
* Never hard-code API keys or tokens.
* Never print secrets in logs.

## 6. Environment Variables

* Use `.env` for local secrets.
* Use `.env.example` for placeholder variable names only (no real secrets).
* Never commit `.env`.

Expected variables include:

* `GEMINI_API_KEY`
* `HF_TOKEN`

If additional provider/model configuration is needed (e.g. model IDs, base URLs), expose them via environment variables as well.

## 7. Image Generation

* Do NOT attempt to run Stable Diffusion locally.
* The development machine does not have a suitable NVIDIA CUDA GPU.
* Use hosted Hugging Face inference.

* Keep image-generation code behind a clean abstraction so the provider/model can be changed later.
* Caller code must not depend on Hugging Face-specific request/response details.

## 8. PDF Export

* Use FPDF.
* The exporter must receive the canonical `Comic` data rather than reconstructing story data independently.
* PDF filenames must be timestamped.
* Output PDFs to `static/exports/`.
* Handle PDF generation failures explicitly with useful errors.

## 9. Frontend

Planned templates:

* `index.html` — story input
* `comic_preview.html` — 5-panel preview
* `export_success.html` — export confirmation/download

* Keep presentation logic in templates and backend logic in Python.
* Do not embed AI, filesystem, or PDF logic in Jinja templates.

## 10. API Design

Support:

* normal form-based comic generation
* JSON/API generation
* image-generation testing
* PDF export

* Use validation and useful error responses.
* Return clear HTTP status codes and messages for invalid input, missing keys, and downstream failures.

## 11. Reliability

Validate AI-generated structured output before using it.

Handle explicitly:

* missing API keys
* API failures
* malformed model output
* image generation failures
* PDF generation failures
* invalid user input

Do not silently swallow errors. Log useful context without leaking secrets.

## 12. Development Workflow

Work incrementally. Do not implement the entire project in one step.

Before moving to the next phase:

* inspect the existing code
* run tests
* run the application when appropriate
* verify the current phase works
* fix failures before continuing

Prefer small, focused changes.

## 13. Git Safety

* Never modify `.env` with real secrets.
* Never commit `.env`.
* Never expose API keys/tokens in source code, logs, tests, documentation, or responses.
* Check `git status` / `git diff` before committing; stage only intended files.

## 14. Code Quality

Prefer:

* clear functions
* type hints
* Pydantic validation
* small modules
* environment-based configuration
* useful error messages

Avoid unnecessary dependencies and unnecessary abstraction.

## 15. V1 Scope

The first version must focus on:

* story input
* 5-panel generation
* AI-generated narration/dialogue
* AI-generated panel images
* comic preview
* PDF export

Do not add authentication, payments, databases, social features, or deployment infrastructure unless explicitly requested.
