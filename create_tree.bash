#!/usr/bin/env bash

cd "$(dirname "$0")"

README="README.md"
TMP_README="$README.tmp"
HEADING="# File Tree"

# Generate file tree excluding bulky/temp files
FILE_TREE=$(find . -type f \
  ! -name ".gstmp" \
  ! -name "*.un~" \
  ! -name "*.swp" \
  ! -name "*~" \
  ! -path "./.git/*" \
  | sed 's|^\./||' | sort | awk '{print "- " $0}')

# Text to insert
TREE_SECTION="$HEADING

$FILE_TREE

### How to regenerate

Run this in the root directory:

\`\`\`bash
./create_tree.bash
\`\`\`"

# If the heading exists, replace from that point on
if grep -q "^$HEADING" "$README"; then
  awk -v heading="$HEADING" -v section="$TREE_SECTION" '
    BEGIN { replaced = 0 }
    {
      if (!replaced && $0 == heading) {
        print section
        replaced = 1
        skip = 1
        next
      }
      if (replaced && /^#/ && $0 != heading) {
        skip = 0
      }
      if (!skip) print
    }
  ' "$README" > "$TMP_README" && mv "$TMP_README" "$README"
else
  echo -e "\n$TREE_SECTION" >> "$README"
fi
