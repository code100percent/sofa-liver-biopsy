#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
release=25.12.00
runtime=".sofa/SOFA_v${release}_Linux"
if [[ -x "$runtime/bin/runSofa" ]]; then exit 0; fi
mkdir -p .sofa
stage=$(mktemp -d .sofa/install.XXXXXX)
trap 'rm -rf -- "$stage"' EXIT
archive="$stage/sofa.zip"
if [[ -n "${SOFA_ARCHIVE:-}" ]]; then
    archive="$SOFA_ARCHIVE"
else
    curl -fL --retry 2 -o "$archive" "https://github.com/sofa-framework/sofa/releases/download/v${release}/SOFA_v${release}_Linux-Python_3.12.zip"
fi
echo "76be4a5d89656a7e8c89bc45a30665aab2a52ba33e77e2a59280b735a9288574  $archive" | sha256sum --check
unzip -q "$archive" -d "$stage"
test -x "$stage/SOFA_v${release}_Linux/bin/runSofa"
mv "$stage/SOFA_v${release}_Linux" "$runtime"
