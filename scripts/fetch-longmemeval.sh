#!/bin/sh
# Downloads the LongMemEval-S cleaned dataset and verifies its SHA256.
set -e

DATA_DIR="$(dirname "$0")/../benchmarks/longmemeval/data"
DEST="$DATA_DIR/longmemeval_s_cleaned.json"
URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
EXPECTED_SHA256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"

verify_sha256() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum > /dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "ERROR: neither sha256sum nor shasum found" >&2
        exit 1
    fi
}

mkdir -p "$DATA_DIR"

if [ -f "$DEST" ]; then
    ACTUAL=$(verify_sha256 "$DEST")
    if [ "$ACTUAL" = "$EXPECTED_SHA256" ]; then
        echo "Dataset already present and hash verified."
        exit 0
    else
        echo "Existing file hash mismatch (got $ACTUAL), re-downloading."
        rm -f "$DEST"
    fi
fi

echo "Downloading $URL ..."
if command -v curl > /dev/null 2>&1; then
    curl -fL --retry 3 -o "$DEST" "$URL"
elif command -v wget > /dev/null 2>&1; then
    wget -O "$DEST" "$URL"
else
    echo "ERROR: neither curl nor wget found" >&2
    exit 1
fi

ACTUAL=$(verify_sha256 "$DEST")
if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: SHA256 mismatch after download." >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  got:      $ACTUAL" >&2
    rm -f "$DEST"
    exit 1
fi

echo "Download complete and hash verified."
