#!/bin/bash
set -e
set -o pipefail

# Script to automate Wikipedia parallel titles extraction
# Usage: ./download_wikipedia_parallel_titles.sh <version_date> <lang1> [lang2] [lang3] ...
# Example: ./download_wikipedia_parallel_titles.sh 20250601 de it hi ko

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <version_date> <lang1> [lang2] [lang3] ..." 1>&2
    echo "Example: $0 20140831 ar de fr" 1>&2
    echo "This will download Wikipedia dumps for the specified languages and extract parallel titles with English" 1>&2
    exit 1
fi

VERSION_DATE=$1
shift  # Remove first argument, remaining are languages
LANGUAGES=("$@")

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/_wiki_parallel_titles"
UTILS_DIR="$SCRIPT_DIR/wikipedia-parallel-titles"

echo "Project root: $PROJECT_ROOT" 1>&2
echo "Data directory: $DATA_DIR" 1>&2
echo "Utils directory: $UTILS_DIR" 1>&2
echo "Version date: $VERSION_DATE" 1>&2
echo "Languages: ${LANGUAGES[*]}" 1>&2

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Function to download file if it doesn't exist
download_file() {
    local url=$1
    local filename=$2
    local filepath="$DATA_DIR/$filename"
    
    if [[ -f "$filepath" ]]; then
        echo "File $filename already exists, skipping download" 1>&2
    else
        echo "Downloading $filename..." 1>&2
        if wget -O "$filepath" "$url"; then
            echo "Successfully downloaded $filename" 1>&2
        else
            echo "Failed to download $filename" 1>&2
            exit 1
        fi
    fi
}

# Function to extract parallel titles for a language
extract_parallel_titles() {
    local lang=$1
    local wiki_prefix="${lang}wiki-${VERSION_DATE}"
    local output_file="$DATA_DIR/${lang}_titles.txt"
    
    echo "Extracting parallel titles for $lang..." 1>&2
    
    # Check if required files exist
    local pages_file="$DATA_DIR/${wiki_prefix}-page.sql.gz"
    local langlinks_file="$DATA_DIR/${wiki_prefix}-langlinks.sql.gz"
    
    if [[ ! -f "$pages_file" ]]; then
        echo "Error: $pages_file not found" 1>&2
        return 1
    fi
    
    if [[ ! -f "$langlinks_file" ]]; then
        echo "Error: $langlinks_file not found" 1>&2
        return 1
    fi
    
    # Change to the utils directory to run build-corpus.sh
    cd "$UTILS_DIR"
    
    # Run build-corpus.sh with adjusted paths
    if ./build-corpus.sh en "$DATA_DIR/${wiki_prefix}" > "$output_file"; then
        echo "Successfully extracted parallel titles for $lang to $output_file" 1>&2
        echo "Number of title pairs: $(wc -l < "$output_file")" 1>&2
    else
        echo "Failed to extract parallel titles for $lang" 1>&2
        return 1
    fi
    
    # Return to original directory
    cd - > /dev/null
}

# Process each language
for lang in "${LANGUAGES[@]}"; do
    echo "Processing language: $lang" 1>&2
    
    # Construct URLs for the Wikipedia dumps
    wiki_prefix="${lang}wiki-${VERSION_DATE}"
    base_url="http://dumps.wikimedia.org/${lang}wiki/${VERSION_DATE}"
    base_url="https://mirror.accum.se/mirror/wikimedia.org/dumps/${lang}wiki/${VERSION_DATE}"
    pages_url="${base_url}/${wiki_prefix}-page.sql.gz"
    langlinks_url="${base_url}/${wiki_prefix}-langlinks.sql.gz"
    
    echo "Downloading files for $lang..." 1>&2
    
    # Download the required files
    download_file "$pages_url" "${wiki_prefix}-page.sql.gz"
    download_file "$langlinks_url" "${wiki_prefix}-langlinks.sql.gz"
    
    # Extract parallel titles
    extract_parallel_titles "$lang"
    
    echo "Completed processing for $lang" 1>&2
    echo "----------------------------------------" 1>&2
done

echo "All languages processed successfully!" 1>&2
echo "Output files are saved in the data directory: $DATA_DIR with the pattern: {lang}_titles.txt" 1>&2
