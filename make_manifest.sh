#!/bin/sh
# Regenerate dist/SHA256SUMS.txt.
#
# The manifest must describe what a CLONE receives, so the file list comes from
# git, not from the working tree. Generating it with `find` picks up untracked
# local files -- IDE config, scratch output, caches -- which then appear in the
# manifest and fail `sha256sum -c` for everyone who clones. That has happened.
#
# Run from the repository root:  sh make_manifest.sh
set -e
cd "$(dirname "$0")"
git ls-files -z \
  | grep -zv '^dist/SHA256SUMS\.txt$' \
  | xargs -0 sha256sum \
  | sed 's/^\([0-9a-f]\{64\}\) \*/\1  /' \
  > dist/SHA256SUMS.txt
echo "dist/SHA256SUMS.txt: $(wc -l < dist/SHA256SUMS.txt) files"
