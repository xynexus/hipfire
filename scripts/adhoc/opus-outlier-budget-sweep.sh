#!/usr/bin/env bash
# Re-run of the per-layer outlier budget sweep (commit d77fa637a) against the
# JOINT mixed_clipsearch selector shipped in 8357081d3.
#
# Two questions off one calibration:
#   ALLOCATION — does any per-layer split beat uniform N=3 at matched bytes?
#     d77fa637a said no. The CPU study (examples/opus_outlier_budget_study.rs)
#     predicts still no: the best possible reallocation is worth 2.39% SSE, and
#     the config d77fa637a tested was aimed the wrong way (down_proj has the
#     LOWEST per-group marginal value, not the highest).
#   BIT RATE  — d77fa637a also found oq4.5++ (N=7) scoring WORSE KLD than N=3
#     while spending MORE bits. That is the signature of a selector that could
#     not use the bits, and it is what the joint selector should overturn.
#
#   scripts/adhoc/opus-outlier-budget-sweep.sh
set -u
cd "$(dirname "$0")/../.."

HF="${HIPFIRE_SWEEP_HF:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
HESS="${HIPFIRE_SWEEP_HESSIAN:-$HOME/.hipfire/calib/qwen3.5-0.8b-full.calib.hfq}"
IM="${HIPFIRE_SWEEP_IMATRIX:-$HOME/.hipfire/imatrix/qwen3.5-0.8b-bf16.imatrix.gguf}"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
W="${HIPFIRE_SWEEP_WORK:-$HOME/.hipfire/format-sweep/qwen3.5-0.8b-p2}"
OUT="$W/opus-outlier-budget-sweep.md"
Q=./target/release/hipfire-quantize
PPL=./target/release/examples/perplexity
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
# Pin the thread count. hipfire-quantize is NOT bit-reproducible across rayon
# thread counts — the LDLQ path's parallel reductions reassociate differently, so
# the same inputs on a busy box and an idle one give different artifacts
# (measured: default vs RAYON_NUM_THREADS=4 → different md5, identical logs).
# A sweep whose configs differ by ~2% must not also vary with machine load.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}"
mkdir -p "$W"

for f in "$HF" "$HESS" "$IM" "$CORPUS" "$Q" "$PPL"; do
    [ -e "$f" ] || {
        echo "missing: $f" >&2
        exit 2
    }
done

# PREFLIGHT: refuse a partial calib.
#
# /srv/hipfire/calib/qwen3.5-0.8b.calib.hfq carries Hessians for down_proj ONLY
# (48 of the model's 488 tensors). `--ldlq` does not fail on that — it logs
# "ldlq: skip <t> (no Hessian entry ...)" per tensor and silently falls back to
# RTN. A sweep run against it applies error feedback to exactly one of the seven
# layer types, which is both not-really-`++` and biased toward the very tensor
# the allocation question is about. That is a silent-wrong-answer machine, so it
# is a hard stop rather than a warning.
echo "[+] preflight: Hessian coverage in $HESS"
COVER=$(./target/release/hipfire inspect "$HESS" --tensors 2>/dev/null |
    grep -oE "[a-z_0-9.]+\.hessian" | sed -E 's/.*\.([a-z_0-9]+)\.hessian/\1/' | sort -u)
echo "$COVER" | sed 's/^/      /'
MISSING=""
for want in down_proj gate_proj up_proj; do
    echo "$COVER" | grep -qx "$want" || MISSING="$MISSING $want"
done
if [ -n "$MISSING" ]; then
    echo "  ABORT: calib lacks Hessians for:$MISSING" >&2
    echo "  --ldlq would silently degrade to RTN for those tensors." >&2
    echo "  Rebuild with: hipfire collect-artifacts --model <hf_dir> \\" >&2
    echo "      --corpus $CORPUS --output <full>.calib.hfq --max-tokens 512" >&2
    exit 3
fi

# name | --format | HIPFIRE_OUTLIERS_BY_LAYER ("-" = unset)
CONFIGS=(
    "uniform3|oq4.25++|-"
    "uniform7|oq4.5++|-"
    "down7rest1|oq4.25++|down_proj:7,default:1"
    "down7o3rest1|oq4.25++|down_proj:7,o_proj:3,default:1"
    "optimum|oq4.25++|q_proj:5,k_proj:6,v_proj:9,o_proj:3,gate_proj:5,up_proj:1,down_proj:2"
)

# hipfire-quantize and the perplexity example are non-daemon GPU binaries: they
# do not self-lock, so the sweep holds the lock for them. `hipfire eval` is the
# documented exception (it loads through the daemon, which locks itself) and is
# run AFTER the release below — wrapping it deadlocks against our own holder.
./target/release/hipfire lock acquire "opus-outlier-sweep" --watch-pid "$$" || {
    echo "lock busy" >&2
    exit 2
}
trap './target/release/hipfire lock release 2>/dev/null || true' EXIT

echo "[+] bf16 reference (KLD needs a higher-precision comparand)"
REF="$W/qwen3.5-0.8b--bf16.hfq"
if [ ! -e "$REF" ]; then
    "$Q" --input "$HF" --output "$REF" --format bf16 >"$W/quant.bf16.log" 2>&1 || {
        echo "  FAIL bf16: $(tail -3 "$W/quant.bf16.log")"
        exit 1
    }
fi
echo "    $(du -h "$REF" | cut -f1)"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name fmt outliers <<<"$cfg"
    out="$W/$name.hfq"
    if [ ! -e "$out" ]; then
        echo "[+] quantize $name (--format $fmt, outliers=${outliers})"
        if [ "$outliers" = "-" ]; then
            unset HIPFIRE_OUTLIERS_BY_LAYER
        else
            export HIPFIRE_OUTLIERS_BY_LAYER="$outliers"
        fi
        "$Q" --input "$HF" --output "$out" --format "$fmt" \
            --hessian "$HESS" --imatrix "$IM" --ldlq --awq \
            >"$W/quant.$name.log" 2>&1 || {
            echo "  FAIL: $(tail -3 "$W/quant.$name.log")"
            continue
        }
        unset HIPFIRE_OUTLIERS_BY_LAYER
    fi
    # Realised bit rate, straight off the artifact — the whole comparison is
    # only meaningful if these land where the configs claim.
    bytes=$(stat -c %s "$out")
    echo "    $name  $(numfmt --to=iec "$bytes")"
done

echo
echo "[+] perplexity (ctx 2048)"
declare -A PP
for cfg in "${CONFIGS[@]}" "bf16|-|-"; do
    IFS='|' read -r name _ _ <<<"$cfg"
    m="$W/$name.hfq"
    [ "$name" = "bf16" ] && m="$REF"
    [ -e "$m" ] || continue
    p=$("$PPL" "$m" "$CORPUS" --ctx 2048 --warmup 8 2>"$W/ppl.$name.err" |
        grep -oiE "PPL:[ ]+[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
    PP[$name]="${p:-NA}"
    echo "    $name ppl=${p:-NA}"
done

./target/release/hipfire lock release 2>/dev/null || true
trap - EXIT

echo
echo "[+] KLD vs bf16 (hipfire eval — NOT lock-wrapped, it locks itself)"
declare -A KL
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name _ _ <<<"$cfg"
    m="$W/$name.hfq"
    [ -e "$m" ] || continue
    ./target/release/hipfire eval "$m" --compare "$REF" --battery quality \
        >"$W/kld.$name.log" 2>&1 || true
    k=$(grep -oiE "mean_kld=[0-9.]+" "$W/kld.$name.log" | grep -oE "[0-9.]+" | tail -1)
    KL[$name]="${k:-NA}"
    echo "    $name kld=${k:-NA}"
done

{
    echo "# Opus per-layer outlier budget — re-run against the joint selector"
    echo
    echo "model: qwen3.5-0.8b · corpus: wikitext2-1024s-2048ctx · ctx 2048"
    echo "hessian: $HESS"
    echo "selector: joint mixed_clipsearch (8357081d3)"
    echo
    printf "| config | bytes | PPL | KLD vs bf16 |\n|---|---:|---:|---:|\n"
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r name _ _ <<<"$cfg"
        b=$([ -e "$W/$name.hfq" ] && numfmt --to=iec "$(stat -c %s "$W/$name.hfq")" || echo NA)
        printf "| %s | %s | %s | %s |\n" "$name" "$b" "${PP[$name]:-NA}" "${KL[$name]:-NA}"
    done
    printf "| bf16 (ref) | %s | %s | 0 |\n" \
        "$(numfmt --to=iec "$(stat -c %s "$REF")")" "${PP[bf16]:-NA}"
    echo
    echo "Prior run (d77fa637a, OLD selector):"
    echo "  uniform N=3        4.2500 b/w  PPL 17.1547  KLD 0.034862  <- best"
    echo "  down=7 rest=1      4.2284 b/w  PPL 17.0964  KLD 0.038551"
    echo "  down=7 o=3 rest=1  4.2371 b/w  PPL 17.1397  KLD 0.039076"
    echo "  uniform N=7 (oq4.5++)          KLD 0.037291  (more bits, worse)"
} | tee "$OUT"
echo "[opus-outlier-sweep] wrote $OUT"
