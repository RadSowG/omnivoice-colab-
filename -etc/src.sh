# The script isn’t being executed—only made executable.
# To actually run it you need to call it:
#   ./-etc/src.sh
# or
#   bash ./-etc/src.sh
#!/bin/bash
#    "src": "bash ./-etc/src.sh",

# Configuration
SOURCE_DIR="./src"
# SOURCE_DIR="."
EXCLUDE_EXTS=("ex5") # Extensions to exclude
OUTPUT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
BASE_NAME="src-OmniVoiceColab"
# BASE_NAME=""
EXTENSION="md"
REPO_ROOT=$(pwd)

# Determine output filename
COUNTER=0
while true; do
    FILE_NAME=$(printf "%s%02d.%s" "$BASE_NAME" "$COUNTER" "$EXTENSION")
    OUTPUT_FILE="$OUTPUT_DIR/$FILE_NAME"
    if [ ! -f "$OUTPUT_FILE" ]; then
        break
    fi
    ((COUNTER++))
done

# Check if src directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: $SOURCE_DIR directory not found in $REPO_ROOT"
    exit 1
fi

# Initialize the output file
# Auto-detect the highest-level folder name (basename of REPO_ROOT)
REPO_NAME=$(basename "$REPO_ROOT")
echo "# ${REPO_NAME} Source Bundle" > "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 1. Directory Structure
echo "## Directory Structure" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
# Use tree with --gitignore to respect .gitignore if available, otherwise just tree src
if tree --gitignore "$SOURCE_DIR" > /dev/null 2>&1; then
    tree "$SOURCE_DIR" --gitignore >> "$OUTPUT_FILE"
else
    tree "$SOURCE_DIR" >> "$OUTPUT_FILE"
fi
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 2. File Contents
echo "## File Contents" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Use git ls-files to get all tracked and non-ignored files in src/
# This ensures .gitignore is respected.
# If not a git repo, fall back to find.
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    files=$(git ls-files "$SOURCE_DIR" --cached --others --exclude-standard)
else
    echo "Warning: Not a git repository. Using 'find' which may not respect .gitignore perfectly."
    files=$(find "$SOURCE_DIR" -type f)
fi

for file in $files; do
    if [ -f "$file" ]; then
        # Check if extension is in exclude list
        extension="${file##*.}"
        skip=false
        for ex in "${EXCLUDE_EXTS[@]}"; do
            if [ "$extension" == "$ex" ]; then
                skip=true
                break
            fi
        done
        [ "$skip" = true ] && continue

        # Determine language for code block
        case "$extension" in
            ts) lang="typescript" ;;
            tsx) lang="tsx" ;;
            js) lang="javascript" ;;
            jsx) lang="jsx" ;;
            json) lang="json" ;;
            css) lang="css" ;;
            md) lang="markdown" ;;
            html) lang="html" ;;
            *) lang="" ;;
        esac

        echo "### File: $file" >> "$OUTPUT_FILE"
        echo '```'$lang >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo '```' >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
done

echo "Successfully created $OUTPUT_FILE"
