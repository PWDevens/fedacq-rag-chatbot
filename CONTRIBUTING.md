# Contributing

This is a personal portfolio project. Issues and pull requests are welcome.

## Development Setup

See the README [Setup & Installation](README.md#setup--installation) section for environment setup (`venv`, `pip install -e .`, `pip install -r requirements.txt`).

## Testing

Run tests with:

```bash
pytest -q
```

Phi-4 model tests are skipped unless `ENABLE_PHI4_TESTS=1` is set. The CI environment runs tests with `RAG_TEST_MODE=1`, which disables expensive model tests while keeping the index smoke test active.

## Style

Keep changes minimal and consistent with the existing codebase. No linter is enforced.

## Commits and PRs

- Write descriptive commit messages that explain the "why".
- One logical change per pull request.

## Large Files

- The Chroma index (`data/chroma/`) is Git LFS-tracked; never rebuild and re-commit it unless instructed.
- The Phi-4 ONNX model is downloaded at setup time (see README). Never commit model weights.
