#!/usr/bin/env bash
# Lit un csv (colonnes device_id, sensor_id, start, end) et le transmet à real_neg_samples.py sous
# la forme d'une liste JSON de dictionnaires {device_id, sensor_id, start, end}.
# Usage: ./run_real_neg_samples.sh [training_data/real_neg_samples/real_neg_samples.csv] [options de real_neg_samples.py]
# Les valeurs ne doivent contenir ni virgule ni guillemet (ids et dates simples).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="${1:-$SCRIPT_DIR/training_data/real_neg_samples/real_neg_samples.csv}"
shift || true

if [ ! -f "$CSV" ]; then
    echo "Csv introuvable : $CSV" >&2
    exit 1
fi

REQUESTS=$(tr -d '\r' < "$CSV" | awk -F',' '
    NR == 1 {
        for (i = 1; i <= NF; i++) { gsub(/^[ \t"]+|[ \t"]+$/, "", $i); col[$i] = i }
        n = split("device_id sensor_id start end", req, " ")
        for (k = 1; k <= n; k++) if (!(req[k] in col)) {
            print "Colonne manquante dans le csv : " req[k] > "/dev/stderr"; exit 1
        }
        printf "["; next
    }
    NF == 0 || $0 ~ /^[ \t,]*$/ { next }
    {
        for (k = 1; k <= n; k++) { v = $(col[req[k]]); gsub(/^[ \t"]+|[ \t"]+$/, "", v); f[k] = v }
        printf "%s{\"device_id\": \"%s\", \"sensor_id\": \"%s\", \"start\": \"%s\", \"end\": \"%s\"}", \
               (first++ ? ", " : ""), f[1], f[2], f[3], f[4]
    }
    END { print "]" }
')

python "$SCRIPT_DIR/real_neg_samples.py" --requests "$REQUESTS" "$@"
