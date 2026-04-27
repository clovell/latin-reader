"""JSON API endpoints."""
from flask import Blueprint, request, jsonify, Response, current_app
from spacy import displacy
import json
import os
import random

bp = Blueprint("api", __name__)


def _displacy_options() -> dict:
    """Shared displaCy rendering options."""
    return {
        "compact": False,
        "bg": "transparent",
        "color": "#e0e0e0",
        "font": "'Inter', sans-serif",
        "distance": 120,
        "offset_x": 80,
        "arrow_stroke": 2,
        "arrow_width": 8,
        "word_spacing": 30,
    }


@bp.route("/analyze", methods=["POST"])
def analyze():
    """Parse Latin text and return DisplaCy SVG visualization + CoNLL-U.

    Accepts JSON: {text: str, gold_tokens?: list}
    Returns JSON: {html: str, conllu: str, tokens: list}
    """
    data = request.get_json()
    text = data.get("text", "").strip()
    gold_tokens = data.get("gold_tokens", None)

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    try:
        if gold_tokens:
            # Phase 3 & 4 for Gold Data
            from latin_reader.latincy_postprocessor.sentence import Token
            from latin_reader.pipeline.chunker import chunk_sentence
            from latin_reader.pipeline.renderer import render_sentence_map
            
            parsed_tokens = []
            for i, t in enumerate(gold_tokens):
                parsed_tokens.append(Token(
                    id=i+1,
                    form=t.get("text", ""),
                    lemma=t.get("lemma", ""),
                    upos=t.get("pos", ""),
                    xpos=t.get("tag", ""),
                    feats={},
                    head=int(t.get("head", 0)),
                    deprel=t.get("dep", "")
                ))
            
            chunk = chunk_sentence(parsed_tokens)
            layout_mode = request.args.get("layout_mode", "clauses_only")
            chunk_svg = render_sentence_map(chunk, parsed_tokens, mode=layout_mode)

            # Manual mode for Perseus Treebank gold standard
            words = [{"text": t["text"], "tag": t["pos"]} for t in gold_tokens]
            arcs = []

            for i, t in enumerate(gold_tokens):
                head_idx = int(t["head"]) - 1
                if head_idx >= 0 and head_idx != i:
                    start = min(i, head_idx)
                    end = max(i, head_idx)
                    dir = "left" if head_idx > i else "right"
                    arcs.append({"start": start, "end": end, "label": t["dep"], "dir": dir})

            ex = {"words": words, "arcs": arcs}
            svg = displacy.render(
                ex,
                style="dep",
                manual=True,
                options=_displacy_options(),
            )
            return jsonify({"html": svg, "chunk_svg": chunk_svg, "tokens": gold_tokens, "conllu": ""})

        else:
            # Standard NLP processing for custom text
            nlp = current_app.extensions["latin_reader_nlp"]

            # Protect Latin abbreviation periods from sentence splitting
            # by temporarily replacing "M." → "M·" etc. before parsing
            protected_text, abbrev_positions = _protect_abbrev_periods(text)
            doc = nlp(protected_text)

            # Generate raw CoNLL-U from spaCy parse
            raw_conllu = _doc_to_conllu(doc)

            # Restore the original periods in the CoNLL-U output
            raw_conllu = _restore_abbrev_periods(raw_conllu)

            # Run the post-processor to harmonize labels
            from latin_reader.pipeline.postprocessor import harmonize_conllu
            harmonized_conllu, changes, report = harmonize_conllu(raw_conllu)

            # Rebuild displaCy data from harmonized CoNLL-U so users
            # see corrected labels by default
            from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
            harmonized_sents = list(read_conllu_string(harmonized_conllu))

            if harmonized_sents:
                # Build displaCy manual data from harmonized sentences
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
                svg = displacy.render(
                    ex,
                    style="dep",
                    manual=True,
                    options=_displacy_options(),
                )
            else:
                # Fallback: render directly from spaCy Doc
                svg = displacy.render(doc, style="dep", options=_displacy_options())

            # Build token data table from harmonized output
            tokens = []
            for sent in harmonized_sents:
                for tok in sent.tokens:
                    head_tok = sent.by_id(tok.head)
                    tokens.append(
                        {
                            "id": tok.id,
                            "text": tok.form,
                            "lemma": tok.lemma,
                            "pos": tok.upos,
                            "tag": tok.xpos,
                            "dep": tok.deprel,
                            "head": head_tok.form if head_tok else tok.form,
                            "head_id": tok.head,
                            "morph": "|".join(f"{k}={v}" for k, v in tok.feats.items()) if tok.feats else "",
                        }
                    )

            # Serialize change log for the response
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

            # Phase 4: Generate Sentence Map SVG from Chunk architecture
            from latin_reader.pipeline.chunker import chunk_sentence
            from latin_reader.pipeline.renderer import render_sentence_map
            
            layout_mode = request.args.get("layout_mode", "full_blocks")
            
            chunk_svgs = []
            for sent in harmonized_sents:
                chunk = chunk_sentence(sent.tokens)
                chunk_svgs.append(render_sentence_map(chunk, sent.tokens, mode=layout_mode))
            combined_chunk_svg = "\n<br>\n".join(chunk_svgs)

            return jsonify({
                "html": svg,
                "chunk_svg": combined_chunk_svg,
                "tokens": tokens,
                "conllu": harmonized_conllu,
                "raw_conllu": raw_conllu,
                "change_count": len(changes),
                "changes": change_log,
            })

    except Exception as e:
        return jsonify({"error": f"Parsing error: {str(e)}"}), 500


@bp.route("/export/conllu", methods=["POST"])
def export_conllu():
    """Parse Latin text and return CoNLL-U formatted output as a download."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    try:
        nlp = current_app.extensions["latin_reader_nlp"]
        doc = nlp(text)
        conllu_text = _doc_to_conllu(doc)
        return Response(conllu_text, mimetype="text/plain")

    except Exception as e:
        return jsonify({"error": f"Export error: {str(e)}"}), 500


def _parse_to_chunks(text: str):
    """Shared helper: parse text → list of (Chunk, tokens) tuples."""
    from latin_reader.pipeline.postprocessor import harmonize_conllu
    from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
    from latin_reader.pipeline.chunker import chunk_sentence

    nlp = current_app.extensions["latin_reader_nlp"]
    protected_text, _ = _protect_abbrev_periods(text)
    doc = nlp(protected_text)
    raw_conllu = _doc_to_conllu(doc)
    raw_conllu = _restore_abbrev_periods(raw_conllu)
    harmonized_conllu, _, _ = harmonize_conllu(raw_conllu)
    sents = list(read_conllu_string(harmonized_conllu))

    chunks_and_tokens = []
    for sent in sents:
        chunk = chunk_sentence(sent.tokens)
        chunks_and_tokens.append((chunk, sent.tokens))
    return chunks_and_tokens


@bp.route("/export/pdf", methods=["POST"])
def export_pdf_route():
    """Parse Latin text and return a PDF with indented clause visualization."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    try:
        from latin_reader.pipeline.exporter import export_pdf
        chunks_and_tokens = _parse_to_chunks(text)
        pdf_bytes = export_pdf(chunks_and_tokens)
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment; filename=sentence_map.pdf"},
        )
    except Exception as e:
        return jsonify({"error": f"PDF export error: {str(e)}"}), 500


@bp.route("/export/docx", methods=["POST"])
def export_docx_route():
    """Parse Latin text and return a DOCX with indented clause visualization."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    try:
        from latin_reader.pipeline.exporter import export_docx
        chunks_and_tokens = _parse_to_chunks(text)
        docx_bytes = export_docx(chunks_and_tokens)
        return Response(
            docx_bytes,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=sentence_map.docx"},
        )
    except Exception as e:
        return jsonify({"error": f"DOCX export error: {str(e)}"}), 500


@bp.route("/random_perseus", methods=["GET"])
def random_perseus():
    """Return a random sentence from the Perseus treebank.

    Optional query param: author=<name>
    Returns JSON: {text: str, gold_conllu: str, tokens: list, author: str}
    """
    from latin_reader.treebanks.perseus import get_random_sentence
    author = request.args.get("author", None)
    result = get_random_sentence(author=author)
    if result is None:
        return jsonify({"error": "No sentences available."}), 404
    return jsonify(result)


@bp.route("/chunk", methods=["POST"])
def chunk_text():
    """Parse CoNLL-U text and return nested chunk trees.

    Accepts JSON: {conllu: str}
    Returns JSON: {chunks: list[dict]}
    """
    from dataclasses import asdict
    from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
    from latin_reader.pipeline.chunker import chunk_sentence

    data = request.get_json()
    conllu_text = data.get("conllu", "").strip()

    if not conllu_text:
        return jsonify({"error": "No CoNLL-U text provided."}), 400

    try:
        sentences = list(read_conllu_string(conllu_text))
        result_chunks = []
        for s in sentences:
            chunk = chunk_sentence(s.tokens)
            result_chunks.append(asdict(chunk))

        return jsonify({"chunks": result_chunks})
    except Exception as e:
        return jsonify({"error": f"Chunking error: {str(e)}"}), 500


@bp.route("/render", methods=["POST"])
def render_map():
    """Parse CoNLL-U text and return SVG sentence map.

    Accepts JSON: {conllu: str}
    Returns JSON: {svg: str}
    """
    from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
    from latin_reader.pipeline.chunker import chunk_sentence
    from latin_reader.pipeline.renderer import render_sentence_map

    data = request.get_json()
    conllu_text = data.get("conllu", "").strip()

    if not conllu_text:
        return jsonify({"error": "No CoNLL-U text provided."}), 400

    try:
        sentences = list(read_conllu_string(conllu_text))
        svgs = []
        for s in sentences:
            chunk = chunk_sentence(s.tokens)
            svg = render_sentence_map(chunk, s.tokens)
            svgs.append(svg)
            
        combined_svg = "\n<br>\n".join(svgs)
        return jsonify({"svg": combined_svg})
    except Exception as e:
        return jsonify({"error": f"Render error: {str(e)}"}), 500


@bp.route("/define", methods=["GET"])
def define_word():
    """Return dictionary placeholder data for a lemma.
    
    Query params: lemma=str, pos=str (optional)
    Returns JSON: {matches: list}
    """
    lemma = request.args.get("lemma", "").strip()
    pos = request.args.get("pos", "").strip().upper()
    
    if not lemma:
        return jsonify({"error": "No lemma provided."}), 400
        
    try:
        # Provide placeholder dictionary data based solely on LatinCy annotations
        results = [{
            "entry": lemma,
            "pos": pos,
            "morph_human": pos,
            "definition": f"[Placeholder definition for {lemma}]"
        }]
        return jsonify({"matches": results})
    except Exception as e:
        return jsonify({"error": f"Lookup error: {str(e)}"}), 500

# Common Latin praenomina abbreviations (single or multi-letter)
_LATIN_ABBREVS = frozenset([
    "A", "C", "D", "G", "K", "L", "M", "N", "P", "Q", "S", "T",
    "Ap", "Cn", "Sp", "Ti", "Tib", "Ser", "Sex",
])

# Sentinel character to replace abbreviation periods before spaCy parsing.
# Middle dot (·) is visually similar but won't trigger sentence splitting.
_ABBREV_SENTINEL = "\u00B7"


def _protect_abbrev_periods(text: str):
    """Replace the period after known Latin abbreviations with a sentinel.

    Returns (protected_text, list_of_char_positions_that_were_replaced).
    """
    import re
    positions = []
    # Match any abbreviation followed by "." and then a space or end
    # Work backwards so positions stay valid
    chars = list(text)
    for m in re.finditer(r'\b(' + '|'.join(re.escape(a) for a in sorted(_LATIN_ABBREVS, key=len, reverse=True)) + r')\.(?=\s|$)', text):
        dot_pos = m.end() - 1  # position of the "."
        chars[dot_pos] = _ABBREV_SENTINEL
        positions.append(dot_pos)
    return "".join(chars), positions


def _restore_abbrev_periods(conllu: str) -> str:
    """Replace sentinel characters back to periods in CoNLL-U output."""
    return conllu.replace(_ABBREV_SENTINEL, ".")


def _doc_to_conllu(doc) -> str:
    """Convert a spaCy Doc to CoNLL-U format."""
    conllu_lines = []

    for sent_i, sent in enumerate(doc.sents):
        conllu_lines.append(f"# sent_id = {sent_i + 1}")
        conllu_lines.append(f"# text = {sent.text}")

        token_list = list(sent)
        tok_to_id = {tok.i: idx + 1 for idx, tok in enumerate(token_list)}

        for local_id, token in enumerate(token_list, start=1):
            form = token.text
            lemma = token.lemma_
            upos = token.pos_
            xpos = token.tag_ if token.tag_ else "_"
            feats = str(token.morph) if str(token.morph) else "_"
            if not feats:
                feats = "_"

            if token.head == token:
                head = "0"
            else:
                head = str(tok_to_id.get(token.head.i, 0))

            deprel = token.dep_ if token.dep_ else "_"
            deps = "_"
            misc = "_"

            conllu_lines.append(
                f"{local_id}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}"
            )

        conllu_lines.append("")

    return "\n".join(conllu_lines)
