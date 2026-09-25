#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_LIST_FILE="$SCRIPT_DIR/site_id_list"
START="2026-01-01 00:00:00"
END="2026-09-09 23:59:59"
FROM_LINE="${1:-1}"
TO_LINE="${2:-10}"

awk -v from="$FROM_LINE" -v to="$TO_LINE" 'NR>=from && NR<=to {print NR","$0}' "$SITE_LIST_FILE" | while IFS=, read -r line_num site_id; do
    site_id="${site_id%$'\r'}"
    [ -z "$site_id" ] && continue
    echo "=== ligne $line_num : site_id=$site_id ==="
    if ! python "$SCRIPT_DIR/main.py" --site_id "$site_id" --start "$START" --end "$END"; then
        echo "!!! Échec pour site_id=$site_id (ligne $line_num), on passe au suivant" >&2
    fi
done
