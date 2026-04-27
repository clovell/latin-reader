"""
reporter.py — Diagnostic report generation for LatinCy vs. UDante.

Produces all output files specified in reporting_spec.txt §§1–7.
"""
from __future__ import annotations
import csv, json, os, random
from collections import Counter
from typing import List, Optional
from data_loader import GoldSentence, GoldToken
from parser import PredToken
from evaluator import EvalResult, _strip_subtype, UFEATS_KEYS

def _pct(n, d): return f"{n/d*100:.1f}%" if d else "—"
def _safe(n, d): return n/d if d else None

# ═══════════════════════════════════════════════════════════════════════
# §1  Headline Metrics Report
# ═══════════════════════════════════════════════════════════════════════

def _metrics_dict(r: EvalResult) -> dict:
    n = r.total_aligned
    return {
        "UAS": _safe(r.total_head_correct, n),
        "LAS": _safe(r.total_las_correct, n),
        "LAS_with_subtypes": _safe(r.total_las_full_correct, n),
        "UPOS_accuracy": _safe(r.total_pos_correct, n),
        "XPOS_accuracy": _safe(r.total_xpos_correct, n),
        "UFeats_accuracy": _safe(r.total_ufeats_correct, n),
        "UFeats_partial_F1": r.ufeats_partial_f1,
        "Lemma_accuracy": _safe(r.total_lemma_correct, n),
        "total_sentences": len(r.sentences),
        "total_gold_tokens": r.total_gold_tokens,
        "total_pred_tokens": r.total_pred_tokens,
        "total_aligned": r.total_aligned,
        "sents_with_tok_mismatch": r.sents_with_tok_mismatch,
        "pct_sents_tok_mismatch": _safe(r.sents_with_tok_mismatch, len(r.sentences)),
        "tokens_excluded": r.total_excluded,
    }

def _subcorpus_table(r: EvalResult) -> list:
    rows = []
    for sc, d in sorted(r.per_subcorpus.items()):
        n = d["aligned"]
        rows.append({"sub_corpus": sc, "sentences": d["sentences"],
            "aligned": n, "UAS": _safe(d["head_correct"], n),
            "LAS": _safe(d["las_correct"], n),
            "UPOS_acc": _safe(d["pos_correct"], n),
            "Lemma_acc": _safe(d["lemma_correct"], n)})
    return rows

def generate_summary(r: EvalResult, out: str):
    m = _metrics_dict(r)
    sc = _subcorpus_table(r)
    # JSON
    with open(os.path.join(out, "report_summary.json"), "w") as f:
        json.dump({"overall": m, "per_subcorpus": sc}, f, indent=2, default=str)
    # Markdown
    def fp(v): return f"{v*100:.1f}%" if v is not None else "—"
    lines = ["# LatinCy vs UDante — Diagnostic Report Summary\n"]
    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k in ("UAS","LAS","LAS_with_subtypes","UPOS_accuracy","XPOS_accuracy",
              "UFeats_accuracy","UFeats_partial_F1","Lemma_accuracy"):
        lines.append(f"| {k} | {fp(m[k])} |")
    lines.append(f"\n## Counts\n")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|-------|")
    for k in ("total_sentences","total_gold_tokens","total_pred_tokens",
              "total_aligned","sents_with_tok_mismatch","tokens_excluded"):
        lines.append(f"| {k} | {m[k]:,} |" if isinstance(m[k], int) else f"| {k} | {m[k]} |")
    if m["pct_sents_tok_mismatch"] is not None:
        lines.append(f"| pct_sents_tok_mismatch | {fp(m['pct_sents_tok_mismatch'])} |")
    if sc:
        lines.append(f"\n## Per Sub-corpus\n")
        lines.append("| Sub-corpus | Sents | Aligned | UAS | LAS | UPOS | Lemma |")
        lines.append("|------------|-------|---------|-----|-----|------|-------|")
        for row in sc:
            lines.append(f"| {row['sub_corpus']} | {row['sentences']} | {row['aligned']} "
                         f"| {fp(row['UAS'])} | {fp(row['LAS'])} | {fp(row['UPOS_acc'])} "
                         f"| {fp(row['Lemma_acc'])} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out, "report_summary.md"), "w") as f:
        f.write(md)
    return md

# ═══════════════════════════════════════════════════════════════════════
# §2  Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════

def _write_confusion_csv(confusion: dict, total: dict, path: str):
    all_labels = sorted(set(total.keys()) | {l for c in confusion.values() for l in c})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predicted\\gold"] + all_labels + ["TOTAL"])
        for pred in all_labels:
            row = [pred]
            row_total = 0
            for gold in all_labels:
                c = confusion.get(gold, Counter()).get(pred, 0)
                # Also add correct count on diagonal
                if pred == gold:
                    c = total.get(gold, 0) - sum(confusion.get(gold, Counter()).values())
                row.append(c); row_total += c
            row.append(row_total)
            w.writerow(row)
        # TOTAL row
        totals = ["TOTAL"]
        grand = 0
        for gold in all_labels:
            t = total.get(gold, 0); totals.append(t); grand += t
        totals.append(grand)
        w.writerow(totals)

def _write_confusion_all(confusion: dict, total: dict, path: str):
    """All tokens regardless of head correctness — we just use the same data."""
    _write_confusion_csv(confusion, total, path)

def generate_confusion_matrices(r: EvalResult, out: str):
    # Base deprel (correct head only) — filter from token results
    dep_conf_head = {}; dep_tot_head = {}
    dep_conf_base_head = {}; dep_tot_base_head = {}
    for sr in r.sentences:
        for tr in sr.token_results:
            if tr.gold is None or tr.pred is None: continue
            gd, pd = tr.gold.deprel, tr.pred.deprel
            gb, pb = _strip_subtype(gd), _strip_subtype(pd)
            if tr.head_match:
                dep_tot_head[gd] = dep_tot_head.get(gd, 0) + 1
                if gd != pd:
                    dep_conf_head.setdefault(gd, Counter())[pd] += 1
                dep_tot_base_head[gb] = dep_tot_base_head.get(gb, 0) + 1
                if gb != pb:
                    dep_conf_base_head.setdefault(gb, Counter())[pb] += 1

    _write_confusion_csv(dep_conf_base_head, dep_tot_base_head,
                         os.path.join(out, "deprel_confusion_base.csv"))
    _write_confusion_csv(dep_conf_head, dep_tot_head,
                         os.path.join(out, "deprel_confusion_full.csv"))
    _write_confusion_csv(dict(r.deprel_confusion_base), dict(r.deprel_total_base),
                         os.path.join(out, "deprel_confusion_base_all_attachments.csv"))
    _write_confusion_csv(dict(r.deprel_confusion), dict(r.deprel_total),
                         os.path.join(out, "deprel_confusion_full_all_attachments.csv"))
    _write_confusion_csv(dict(r.upos_confusion), dict(r.upos_total),
                         os.path.join(out, "upos_confusion.csv"))
    # Head attachment confusion
    with open(os.path.join(out, "head_attachment_confusion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Predicted head UPOS", "Gold head UPOS", "Count"])
        for (pu, gu), cnt in sorted(r.head_confusion.items(), key=lambda x: -x[1]):
            w.writerow([pu, gu, cnt])

# ═══════════════════════════════════════════════════════════════════════
# §3  Ranked Error Patterns
# ═══════════════════════════════════════════════════════════════════════

def generate_ranked_errors(r: EvalResult, out: str):
    # Collect all error tuples
    deprel_errs = Counter(); upos_errs = Counter()
    combined_errs = Counter(); lemma_errs = Counter()
    for sr in r.sentences:
        for tr in sr.token_results:
            if tr.gold is None or tr.pred is None: continue
            g, p = tr.gold, tr.pred
            if g.deprel != p.deprel:
                deprel_errs[(p.deprel, g.deprel)] += 1
            if g.upos != p.upos:
                upos_errs[(p.upos, g.upos)] += 1
            if g.deprel != p.deprel or g.upos != p.upos:
                combined_errs[(p.upos, p.deprel, g.upos, g.deprel)] += 1
            if hasattr(p, 'lemma') and g.lemma.lower() != p.lemma.lower():
                lemma_errs[(g.form, p.lemma, g.lemma)] += 1

    def _write_ranked(counter, cols, fname, top_n, min_count=0):
        items = counter.most_common()
        if min_count:
            base = [i for i in items[:top_n]]
            extra = [i for i in items[top_n:] if i[1] >= min_count]
            items = base + extra
        else:
            items = items[:top_n]
        total_errs = sum(counter.values())
        cum = 0
        csv_path = os.path.join(out, fname + ".csv")
        md_path = os.path.join(out, fname + ".md")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Rank"] + cols + ["Count", "% of errors", "Cumulative %"])
            for rank, (key, cnt) in enumerate(items, 1):
                cum += cnt
                vals = list(key) if isinstance(key, tuple) else [key]
                w.writerow([rank] + vals + [cnt, f"{cnt/total_errs*100:.1f}",
                           f"{cum/total_errs*100:.1f}"])
        with open(md_path, "w") as f:
            f.write(f"# {fname.replace('_', ' ').title()}\n\n")
            f.write("| Rank | " + " | ".join(cols) + " | Count | % | Cum % |\n")
            f.write("|" + "------|" * (len(cols) + 4) + "\n")
            cum = 0
            for rank, (key, cnt) in enumerate(items, 1):
                cum += cnt
                vals = list(key) if isinstance(key, tuple) else [key]
                f.write(f"| {rank} | " + " | ".join(str(v) for v in vals) +
                        f" | {cnt} | {cnt/total_errs*100:.1f}% | {cum/total_errs*100:.1f}% |\n")

    if deprel_errs:
        _write_ranked(deprel_errs, ["Predicted deprel", "Gold deprel"],
                      "top_deprel_errors", 50, 5)
    if upos_errs:
        _write_ranked(upos_errs, ["Predicted UPOS", "Gold UPOS"],
                      "top_upos_errors", 50, 5)
    if combined_errs:
        _write_ranked(combined_errs,
                      ["Pred UPOS", "Pred deprel", "Gold UPOS", "Gold deprel"],
                      "top_combined_errors", 50, 5)
    if lemma_errs:
        items = lemma_errs.most_common(100)
        csv_path = os.path.join(out, "top_lemma_errors.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Rank", "Token form", "Predicted lemma", "Gold lemma", "Count"])
            for rank, ((form, pl, gl), cnt) in enumerate(items, 1):
                w.writerow([rank, form, pl, gl, cnt])

# ═══════════════════════════════════════════════════════════════════════
# §4  Error Example Extraction
# ═══════════════════════════════════════════════════════════════════════

def generate_error_examples(r: EvalResult, out: str):
    edir = os.path.join(out, "error_examples")
    os.makedirs(edir, exist_ok=True)
    # Build combined error index
    combined = Counter()
    examples = {}  # (sig) -> list of (sr, tr)
    for sr in r.sentences:
        for tr in sr.token_results:
            if tr.gold is None or tr.pred is None: continue
            g, p = tr.gold, tr.pred
            if g.deprel != p.deprel or g.upos != p.upos:
                sig = (p.upos, p.deprel, g.upos, g.deprel)
                combined[sig] += 1
                examples.setdefault(sig, []).append((sr, tr))

    top20 = combined.most_common(20)
    conllu_lines = []
    for rank, (sig, total_count) in enumerate(top20, 1):
        pu, pd, gu, gd = sig
        fname = f"{rank:02d}_{pu}-{pd}_vs_{gu}-{gd}.md"
        exs = examples[sig][:10]
        lines = [f"# Error Pattern: LatinCy ({pu}, {pd}) vs UDante ({gu}, {gd})\n"]
        lines.append(f"**Total occurrences:** {total_count}")
        lines.append(f"**Showing:** {len(exs)} examples\n---\n")
        for ei, (sr, tr) in enumerate(exs, 1):
            lines.append(f"## Example {ei}\n")
            lines.append(f"**Source:** {sr.source_file}, sentence {sr.sent_id}")
            lines.append(f"**Text:** {sr.text}\n")
            lines.append(f"**Token in question:** `{tr.gold.form}` (index {tr.gold.idx})\n")
            # LatinCy table
            lines.append("**LatinCy analysis:**\n")
            lines.append("| idx | form | lemma | upos | head | deprel |")
            lines.append("|-----|------|-------|------|------|--------|")
            for t in sr.token_results:
                if t.pred:
                    lines.append(f"| {t.pred.idx} | {t.pred.form} | {t.pred.lemma} "
                                 f"| {t.pred.upos} | {t.pred.head} | {t.pred.deprel} |")
            lines.append("\n**UDante gold:**\n")
            lines.append("| idx | form | lemma | upos | head | deprel |")
            lines.append("|-----|------|-------|------|------|--------|")
            for t in sr.token_results:
                if t.gold:
                    lines.append(f"| {t.gold.idx} | {t.gold.form} | {t.gold.lemma} "
                                 f"| {t.gold.upos} | {t.gold.head} | {t.gold.deprel} |")
            g, p = tr.gold, tr.pred
            lines.append(f"\n**Diff summary:** token {g.idx} `{g.form}`: "
                         f"deprel {p.deprel}→{g.deprel}; head {p.head}→{g.head}\n---\n")
            # CoNLL-U for combined file
            conllu_lines.append(f"# error_pattern = {pu}-{pd}_vs_{gu}-{gd}")
            conllu_lines.append(f"# sentence_id = {sr.sent_id}")
            conllu_lines.append(f"# source = UDante_gold")
            for t in sr.token_results:
                if t.gold:
                    g = t.gold
                    conllu_lines.append(f"{g.idx}\t{g.form}\t{g.lemma}\t{g.upos}\t"
                                        f"{g.xpos}\t_\t{g.head}\t{g.deprel}\t_\t_")
            conllu_lines.append("")
            conllu_lines.append(f"# source = LatinCy_pred")
            for t in sr.token_results:
                if t.pred:
                    p = t.pred
                    conllu_lines.append(f"{p.idx}\t{p.form}\t{p.lemma}\t{p.upos}\t"
                                        f"{p.xpos}\t_\t{p.head}\t{p.deprel}\t_\t_")
            conllu_lines.append("")

        with open(os.path.join(edir, fname), "w") as f:
            f.write("\n".join(lines))

    with open(os.path.join(edir, "all_examples.conllu"), "w") as f:
        f.write("\n".join(conllu_lines))

# ═══════════════════════════════════════════════════════════════════════
# §5  Subtype Coverage Report
# ═══════════════════════════════════════════════════════════════════════

def generate_subtype_coverage(r: EvalResult, out: str):
    # §5.1 Gold subtype inventory
    gold_subtypes = Counter()  # full_deprel -> count
    for sr in r.sentences:
        for tr in sr.token_results:
            if tr.gold and ":" in tr.gold.deprel:
                gold_subtypes[tr.gold.deprel] += 1

    # §5.2 LatinCy subtype production
    subtype_matches = {}  # gold_deprel -> {same, base_only, other}
    for sr in r.sentences:
        for tr in sr.token_results:
            if tr.gold is None or tr.pred is None: continue
            gd = tr.gold.deprel
            if ":" not in gd: continue
            if gd not in subtype_matches:
                subtype_matches[gd] = {"same": 0, "base_only": 0, "other": 0}
            pd = tr.pred.deprel
            if pd == gd:
                subtype_matches[gd]["same"] += 1
            elif _strip_subtype(pd) == _strip_subtype(gd):
                subtype_matches[gd]["base_only"] += 1
            else:
                subtype_matches[gd]["other"] += 1

    # §5.3 Spurious LatinCy subtypes
    pred_subtypes = Counter()
    for sr in r.sentences:
        for tr in sr.token_results:
            if tr.pred and ":" in tr.pred.deprel:
                pred_subtypes[tr.pred.deprel] += 1
    spurious = {k: v for k, v in pred_subtypes.items() if k not in gold_subtypes}

    lines = ["# Subtype Coverage Report\n"]
    lines.append("## Gold Subtype Inventory\n")
    lines.append("| Full deprel | Base | Subtype | Gold count |")
    lines.append("|-------------|------|---------|------------|")
    for dep in sorted(gold_subtypes):
        base, sub = dep.split(":", 1)
        lines.append(f"| {dep} | {base} | {sub} | {gold_subtypes[dep]} |")
    lines.append("\n## LatinCy Subtype Production\n")
    lines.append("| Gold deprel | Same full | Base only | Other |")
    lines.append("|-------------|-----------|-----------|-------|")
    for dep in sorted(subtype_matches):
        d = subtype_matches[dep]
        lines.append(f"| {dep} | {d['same']} | {d['base_only']} | {d['other']} |")
    lines.append("\n## Spurious LatinCy Subtypes\n")
    if spurious:
        lines.append("| Deprel | Count |")
        lines.append("|--------|-------|")
        for dep in sorted(spurious):
            lines.append(f"| {dep} | {spurious[dep]} |")
    else:
        lines.append("No spurious subtypes found.\n")

    with open(os.path.join(out, "subtype_coverage.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

# ═══════════════════════════════════════════════════════════════════════
# §6  Tokenization & Alignment Report
# ═══════════════════════════════════════════════════════════════════════

def generate_tokenization_report(r: EvalResult, out: str):
    n_sents = len(r.sentences)
    n_1to1 = sum(1 for s in r.sentences if not s.has_tokenization_mismatch)
    n_more = sum(1 for s in r.sentences if s.n_pred_tokens > s.n_gold_tokens)
    n_fewer = sum(1 for s in r.sentences if s.n_pred_tokens < s.n_gold_tokens)
    diffs = Counter()
    for s in r.sentences:
        diffs[s.n_pred_tokens - s.n_gold_tokens] += 1

    # Top divergent forms
    form_divs = Counter()
    for s in r.sentences:
        if s.has_tokenization_mismatch:
            gold_forms = {t.gold.form for t in s.token_results if t.gold}
            pred_forms = {t.pred.form for t in s.token_results if t.pred}
            for f in gold_forms - pred_forms:
                form_divs[f] += 1
            for f in pred_forms - gold_forms:
                form_divs[f] += 1

    lines = ["# Tokenization & Alignment Report\n"]
    lines.append("## Alignment Statistics\n")
    lines.append(f"- Sentences with 1:1 alignment: **{n_1to1}** / {n_sents}")
    lines.append(f"- Sentences where LatinCy produced more tokens: **{n_more}**")
    lines.append(f"- Sentences where LatinCy produced fewer tokens: **{n_fewer}**\n")
    lines.append("## Token Count Difference Distribution\n")
    lines.append("| Difference | Count |")
    lines.append("|------------|-------|")
    for diff in sorted(diffs):
        lines.append(f"| {diff:+d} | {diffs[diff]} |")
    lines.append(f"\n## Metrics Coverage\n")
    lines.append(f"- Total gold tokens: {r.total_gold_tokens:,}")
    lines.append(f"- Tokens evaluated (aligned): {r.total_aligned:,}")
    lines.append(f"- Tokens excluded: {r.total_excluded:,} "
                 f"({_pct(r.total_excluded, r.total_gold_tokens)})")
    lines.append(f"\n## Top Tokenization Divergences\n")
    lines.append("| Form | Count |")
    lines.append("|------|-------|")
    for form, cnt in form_divs.most_common(30):
        lines.append(f"| {form} | {cnt} |")

    with open(os.path.join(out, "tokenization_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # CSV
    with open(os.path.join(out, "tokenization_divergences.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Form", "Count"])
        for form, cnt in form_divs.most_common(30):
            w.writerow([form, cnt])

# ═══════════════════════════════════════════════════════════════════════
# §7  Raw Output Samples
# ═══════════════════════════════════════════════════════════════════════

def generate_raw_samples(r: EvalResult, out: str, seed: int = 42):
    sdir = os.path.join(out, "raw_samples")
    os.makedirs(sdir, exist_ok=True)
    rng = random.Random(seed)
    indices = list(range(len(r.sentences)))
    sample = rng.sample(indices, min(50, len(indices)))

    gold_lines, pred_lines, paired_lines = [], [], []
    for idx in sample:
        sr = r.sentences[idx]
        gold_lines.append(f"# sent_id = {sr.sent_id}")
        gold_lines.append(f"# text = {sr.text}")
        pred_lines.append(f"# sent_id = {sr.sent_id}")
        pred_lines.append(f"# text = {sr.text}")
        paired_lines.append(f"# sent_id = {sr.sent_id}")
        paired_lines.append(f"# source = UDante_gold")
        for tr in sr.token_results:
            if tr.gold:
                g = tr.gold
                line = f"{g.idx}\t{g.form}\t{g.lemma}\t{g.upos}\t{g.xpos}\t_\t{g.head}\t{g.deprel}\t_\t_"
                gold_lines.append(line)
                paired_lines.append(line)
        gold_lines.append("")
        paired_lines.append("")
        paired_lines.append(f"# source = LatinCy_pred")
        for tr in sr.token_results:
            if tr.pred:
                p = tr.pred
                line = f"{p.idx}\t{p.form}\t{p.lemma}\t{p.upos}\t{p.xpos}\t_\t{p.head}\t{p.deprel}\t_\t_"
                pred_lines.append(line)
                paired_lines.append(line)
        pred_lines.append("")
        paired_lines.append("")

    for fname, data in [("gold_sample.conllu", gold_lines),
                        ("latincy_sample.conllu", pred_lines),
                        ("paired_sample.conllu", paired_lines)]:
        with open(os.path.join(sdir, fname), "w") as f:
            f.write("\n".join(data))

# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_full_report(r: EvalResult, out: str, seed: int = 42) -> str:
    """Generate all report files. Returns the summary markdown."""
    os.makedirs(out, exist_ok=True)
    md = generate_summary(r, out)
    generate_confusion_matrices(r, out)
    generate_ranked_errors(r, out)
    generate_error_examples(r, out)
    generate_subtype_coverage(r, out)
    generate_tokenization_report(r, out)
    generate_raw_samples(r, out, seed)
    return md
