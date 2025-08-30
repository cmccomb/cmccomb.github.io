#!/usr/bin/env bash

# Serve the Jekyll site locally.
#
# Requires:
#   - bundler
#   - jekyll
#
# Usage:
#   ./_scripts/serve.sh
#   JEKYLL_ENV=production ./_scripts/serve.sh
#
# Environment variables:
#   JEKYLL_ENV  Jekyll environment (default: development). The script forwards
#               this value to Jekyll unchanged.
#
# Exit codes:
#   0   server started successfully
#   >0  bundler or jekyll returned a non-zero status

set -euo pipefail

bundle exec jekyll serve
