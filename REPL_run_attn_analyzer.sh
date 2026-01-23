#!/bin/bash


set -e  # Exit on any error

echo "Starting..."

usage() {
	echo "Usage: $0 [--sizes \"128 64\"|--sizes=128,64|-s \"128\"] <ll> [ll]" >&2
	echo ""
	echo "Args:"
	echo "  --sizes, -s     Basket sizes (space- or comma-separated). Default: 128"
	echo "  <ll> [ll]       One or two ISO-639-1 language codes, e.g., en or en de" 
	echo ""
	echo "Examples:"
	echo "  $0 en"
	echo "  $0 --sizes 128,64 en"
	echo "  $0 -s \"256 128\" en de"
}

# Parse CLI options and positional language args
LANGS=()
sizes_override=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--sizes)
			if [[ -z "${2:-}" ]]; then
				echo "Error: --sizes requires a value" >&2; usage; exit 1
			fi
			sizes_override="${2//,/ }"
			shift 2
			;;
		--sizes=*)
			val="${1#*=}"
			sizes_override="${val//,/ }"
			shift
			;;
		-s)
			if [[ -z "${2:-}" ]]; then
				echo "Error: -s requires a value" >&2; usage; exit 1
			fi
			sizes_override="${2//,/ }"
			shift 2
			;;
		-*)
			echo "Error: Unknown option '$1'" >&2; usage; exit 1
			;;
		*)
			LANGS+=("$1")
			shift
			;;
	esac
done

# Validate languages (must be 1 or 2 two-letter lowercase codes)
if [ ${#LANGS[@]} -lt 1 ] || [ ${#LANGS[@]} -gt 2 ]; then
	echo "Error: Provide one or two language codes" >&2; usage; exit 1
fi
for arg in "${LANGS[@]}"; do
	case "$arg" in
		[a-z][a-z]) ;;
		*) echo "Error: language codes must be two lowercase letters (e.g., en, de, hi). Got: '$arg'" >&2; exit 1 ;;
	esac
done

if [ ${#LANGS[@]} -eq 1 ]; then
	LANG_SUFFIX="${LANGS[0]}"
else
	LANG_SUFFIX="${LANGS[0]}_${LANGS[1]}"
fi

CONFIG_DIR="config/Alibaba_mGTE/wiki_parallel"

# Basket sizes: prefer --sizes/-s CLI; else env var BASKET_SIZES (space- or comma-separated); else default 128
if [ -n "$sizes_override" ]; then
	sizes_str="$sizes_override"
elif [ -n "${BASKET_SIZES:-}" ]; then
	sizes_str="${BASKET_SIZES//,/ }"
else
	sizes_str="128"
fi
# Build array from sizes_str
read -r -a BASKET_SIZES_ARR <<< "$sizes_str"
# Validate basket sizes are positive integers
for sz in "${BASKET_SIZES_ARR[@]}"; do
	if ! [[ "$sz" =~ ^[0-9]+$ ]]; then
		echo "Error: basket sizes must be positive integers. Got: '$sz'" >&2
		exit 1
	fi
done

echo "Basket sizes: ${BASKET_SIZES_ARR[*]}"

# Track failed runs (store: config path + CLI args)
FAILED_RUNS=()

# Helper to run a single analyzer call with guarded error handling
# Args:
#   $1  - label (for logging)
#   $@  - remaining are the CLI args array passed to the analyzer
run_analyzer_guarded() {
	local label="$1"
	shift
	local args=("$@")

	echo "$label"
	if ! poetry run python -m locobench.attention.attention_analyzer "${args[@]}"; then
		echo "✗ Failed: ${label}" >&2
		# Record config + full CLI for later summary
		FAILED_RUNS+=("${CONFIG_PATH} :: ${args[*]}")
	fi
}

for i in 1 2 3; do
	CONFIG_PATH="${CONFIG_DIR}/embedding_config_wiki_parallel_${i}_${LANG_SUFFIX}.json"

	for BASKET in "${BASKET_SIZES_ARR[@]}"; do
		# Common arguments for this basket size
		COMMON_ARGS=(--config "${CONFIG_PATH}" --analysis_mode baskets --basket_size "${BASKET}" --batch_size 1)

		# 1) Exclude first and last token (default exclusions) with only_from_first_token
		run_analyzer_guarded "${i}; bs=${BASKET}; excl. 1st and last token" \
			"${COMMON_ARGS[@]}" \
			--only_from_first_token

		# # 2) Include first token
		# run_analyzer_guarded "${i}; bs=${BASKET}; incl. 1st token" \
		# 	"${COMMON_ARGS[@]}" \
		# 	--only_from_first_token \
		# 	--no-exclude_first_token

		# # 3) Include first and last token
		# run_analyzer_guarded "${i}; bs=${BASKET}; incl. 1st token and last token" \
		# 	"${COMMON_ARGS[@]}" \
		# 	--only_from_first_token \
		# 	--no-exclude_first_token \
		# 	--no-exclude_last_token
	done

	# echo "${i}; all tokens"
	# poetry run python -m locobench.attention.attention_analyzer --config "${CONFIG_PATH}" --analysis_mode baskets --basket_size 128 --batch_size 1 --exclude_incoming
done

# Summary of failures
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
	echo ""
	echo "Summary: ${#FAILED_RUNS[@]} failed run(s):"
	for entry in "${FAILED_RUNS[@]}"; do
		echo "  - ${entry}"
	done
else
	echo "All runs succeeded."
fi