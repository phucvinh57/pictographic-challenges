# Pictographic challenges

For the detail of challenges, please visit https://pictographic.ai/career.

Here is the explanation of each challenges
- [Challenge 1: raster outline to SVG](src/challenge_1/README.md).
- Challenge 2: extract centerline. It has not done yet.
## Presequisite

### Git LFS

Test images are stored in Git LFS. Please install `git-lfs` to pull them into your local repository.


### Set up project

This project is managed with [`uv`](https://docs.astral.sh/uv/). Install `uv`
first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this directory, install the project and its dependencies into a
local virtual environment (`.venv`):

```bash
uv sync
```
## How to run

Active `venv` first:

```bash
source .venv/bin/activate
```

## Challenge 1

```bash
uv run challenge_1
```

Output will be under `./output/challenge_1`.
