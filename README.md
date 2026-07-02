# Word frequency calibration using external information and topic models

Code accompanying our poster at **AMLaP 2026** (2–4 September 2026, Saarland University).

**Authors:** Rastislav Hronský & Emmanuel Keuleers — Department of Computational Cognitive Science, Tilburg University

---

## Overview

Word frequency is one of the strongest predictors of lexical decision latencies, but frequency estimates are highly corpus-dependent. We propose a method to calibrate corpus topic mixtures using small, independently obtained *seed* signals — without optimizing directly on the behavioral data being predicted.

Using **Latent Dirichlet Allocation (LDA)**, we decompose two English corpora (Wikipedia, ~509 MB; movie subtitles, ~466 MB) into topics over a vocabulary drawn from the British Lexicon Project (BLP). We then search for topic-mixing proportions that best match two external seeds:

- **Topviews** — the 10 most-visited Wikipedia pages of 2025
- **SWOW** — word association norms from the Small World of Words project

The resulting topic-weighted frequencies are evaluated against lexical decision latencies from the BLP, compared against a baseline (raw corpus frequencies) and a skyline (topic mixture optimized directly on LD data).

## Key finding

SWOW-based calibration consistently improved frequency–RT fit, approaching the skyline $R^2$ with as few as 128–512 word association cues. Topviews seeds also improved fit, though less consistently (pages about movies worked best). Results suggest corpus frequencies can be calibrated from small, easily obtainable external signals.

## Repository structure

```
analysis.ipynb      # Main analysis notebook
prepare_data.ipynb  # Data preparation
topic_utils.py      # LDA utilities
util.py             # General utilities
results.pdf         # Main results figure
results_swow.csv    # SWOW simulation results
results_topviews.csv# Topviews simulation results
data/               # Corpora and BLP stimuli (not included)
AMLaP2026-LaTeX-Template/  # Abstract source (LaTeX)
```

## Requirements

```
pip install -r requirements.txt
```

Dependencies: `gensim`, `numpy`, `pandas`, `matplotlib`, `seaborn`

## Data

The corpora (Wikipedia, OpenSubtitles), BLP stimuli, and SWOW norms are not included in this repository due to size and licensing. See the notebook for expected paths and formats.

## Citation

If you use this code, please cite our abstract:

> Hronský, R. & Keuleers, E. (2026). *Word frequency calibration using external information and topic models.* Poster presented at AMLaP 2026, Saarland University.
