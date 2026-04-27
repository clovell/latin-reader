"""Developer-mode routes."""
from flask import Blueprint, render_template, request, jsonify, current_app
from spacy import displacy

bp = Blueprint("dev", __name__)


@bp.route("/")
def dev_home():
    """Dev-mode home page — shows side-by-side raw vs harmonized parse."""
    text = request.args.get("text", "")
    return render_template("dev_raw.html", text=text)


@bp.route("/compare", methods=["POST"])
def compare():
    """Parse text and return both raw and harmonized DisplaCy SVGs.

    Accepts JSON: {text: str}
    Returns JSON: {
        raw_html: str,
        harmonized_html: str,
        raw_conllu: str,
        harmonized_conllu: str,
        changes: list,
        change_count: int,
    }
    """
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    try:
        from latin_reader.routes.api import _displacy_options, _doc_to_conllu
        from latin_reader.pipeline.postprocessor import harmonize_conllu
        from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string

        nlp = current_app.extensions["latin_reader_nlp"]
        doc = nlp(text)

        # 1. Raw SVG directly from spaCy Doc
        raw_svg = displacy.render(doc, style="dep", options=_displacy_options())

        # 2. Generate raw CoNLL-U
        raw_conllu = _doc_to_conllu(doc)

        # 3. Run post-processor
        harmonized_conllu, changes, report = harmonize_conllu(raw_conllu)

        # 4. Build harmonized SVG from post-processed CoNLL-U
        harmonized_sents = list(read_conllu_string(harmonized_conllu))
        words = []
        arcs = []
        offset = 0
        for sent in harmonized_sents:
            for tok in sent.tokens:
                words.append({"text": tok.form, "tag": tok.upos})
            for tok in sent.tokens:
                if tok.head > 0:
                    head_idx = offset + tok.head - 1
                    dep_idx = offset + tok.id - 1
                    start = min(head_idx, dep_idx)
                    end = max(head_idx, dep_idx)
                    direction = "left" if head_idx > dep_idx else "right"
                    arcs.append({
                        "start": start, "end": end,
                        "label": tok.deprel, "dir": direction,
                    })
            offset += len(sent.tokens)

        ex = {"words": words, "arcs": arcs}
        harmonized_svg = displacy.render(
            ex, style="dep", manual=True, options=_displacy_options(),
        )

        change_log = [
            {
                "token": c.token_form,
                "field": c.field,
                "old": c.old_value,
                "new": c.new_value,
                "rule": c.rule_name,
            }
            for c in changes
        ]

        return jsonify({
            "raw_html": raw_svg,
            "harmonized_html": harmonized_svg,
            "raw_conllu": raw_conllu,
            "harmonized_conllu": harmonized_conllu,
            "changes": change_log,
            "change_count": len(changes),
            "report_summary": report.summary(),
        })

    except Exception as e:
        return jsonify({"error": f"Comparison error: {str(e)}"}), 500
