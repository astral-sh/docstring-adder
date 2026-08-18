# Contributing

Install Node.js and npm before running `uv run prek run --all-files`; CI uses
Node.js 24. The local Prettier hook installs the checked-in `package-lock.json`
with `npm ci --ignore-scripts` before formatting files.
