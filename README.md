# Arxiv Reviewer

A toolkit for fetching arXiv preprints, generating structured reviews with
OpenAI models, and optionally uploading the results to Arweave.  The project
can now be used both as a library and as a command line tool.

Set the environment variable `SCORE_LAYERS` to dampen overly generous quality
scores.  Each additional layer further reduces the reported score.  The default
is `2`.

## Library usage

```python
from pathlib import Path
from arxiv_reviewer import process_papers

process_papers(
    category="cs.LO",
    num_papers=1,
    max_results=5,
    output_dir=Path("out"),
    openai_key="YOUR_KEY",
    upload=False,
    bundlr_wallet=None,
    bundlr_currency="arweave",
    generate_graphql=False,
    serve_graphql=False,
    benchmark=False,
)
```

## Command line

The CLI exposes additional options for selecting the OpenAI models used in the
pipeline.  Display all available flags with:

```
python -m arxiv_reviewer --help
```
