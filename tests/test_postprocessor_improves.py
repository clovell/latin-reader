"""Tests for Phase 2: Post-processor integration.

Verifies that:
1. The postprocessor wrapper (harmonize_conllu) runs without error.
2. read_conllu_string / write_conllu_string round-trip correctly.
3. The /api/analyze endpoint returns post-processor data fields.
4. The /dev/compare endpoint returns both raw and harmonized parses.
5. Known improvement cases: labels are corrected after post-processing.
"""
import json
import pytest


# ---------------------------------------------------------------------------
# Unit tests: conllu_io string helpers
# ---------------------------------------------------------------------------

def test_read_write_conllu_string_roundtrip():
    """read_conllu_string → write_conllu_string should preserve content."""
    from latin_reader.latincy_postprocessor.conllu_io import (
        read_conllu_string,
        write_conllu_string,
    )

    sample = (
        "# sent_id = 1\n"
        "# text = Gallia est omnis divisa.\n"
        "1\tGallia\tGallia\tPROPN\tPROPN\t_\t4\tnsubj:pass\t_\t_\n"
        "2\test\tsum\tAUX\tAUX\t_\t4\taux:pass\t_\t_\n"
        "3\tomnis\tomnis\tDET\tDET\t_\t1\tdet\t_\t_\n"
        "4\tdivisa\tdivido\tVERB\tVERB\t_\t0\tROOT\t_\tSpaceAfter=No\n"
        "5\t.\t.\tPUNCT\tPUNCT\t_\t4\tpunct\t_\t_\n"
        "\n"
    )
    sentences = list(read_conllu_string(sample))
    assert len(sentences) == 1
    assert len(sentences[0].tokens) == 5
    assert sentences[0].tokens[0].form == "Gallia"
    assert sentences[0].tokens[0].deprel == "nsubj:pass"

    # Round-trip
    output = write_conllu_string(sentences)
    assert "Gallia" in output
    assert "nsubj:pass" in output
    sentences2 = list(read_conllu_string(output))
    assert len(sentences2) == 1
    assert len(sentences2[0].tokens) == 5


def test_read_conllu_string_empty():
    """Empty string should yield no sentences."""
    from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
    assert list(read_conllu_string("")) == []


# ---------------------------------------------------------------------------
# Unit tests: postprocessor wrapper
# ---------------------------------------------------------------------------

def test_harmonize_conllu_runs():
    """harmonize_conllu should run without error on valid CoNLL-U input."""
    from latin_reader.pipeline.postprocessor import harmonize_conllu

    sample = (
        "# sent_id = 1\n"
        "# text = Roma pulchra est.\n"
        "1\tRoma\tRoma\tPROPN\tPROPN\t_\t3\tnsubj\t_\t_\n"
        "2\tpulchra\tpulcher\tADJ\tADJ\tCase=Nom|Gender=Fem|Number=Sing\t1\tamod\t_\t_\n"
        "3\test\tsum\tAUX\tAUX\t_\t0\tROOT\t_\tSpaceAfter=No\n"
        "4\t.\t.\tPUNCT\tPUNCT\t_\t3\tpunct\t_\t_\n"
        "\n"
    )
    result_conllu, changes, report = harmonize_conllu(sample)
    assert isinstance(result_conllu, str)
    assert isinstance(changes, list)
    assert "Roma" in result_conllu


def test_harmonize_conllu_detects_obl_agent():
    """Post-processor should relabel obl → obl:agent for a/ab + ablative + passive verb."""
    from latin_reader.pipeline.postprocessor import harmonize_conllu

    # "dictum est a Cicerone" — "a Cicerone" should be obl:agent, not plain obl
    sample = (
        "# sent_id = 1\n"
        "# text = Dictum est a Cicerone.\n"
        "1\tDictum\tdico\tVERB\tVERB\tCase=Nom|Gender=Neut|Number=Sing|Tense=Past|VerbForm=Part|Voice=Pass\t0\tROOT\t_\t_\n"
        "2\test\tsum\tAUX\tAUX\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t1\taux:pass\t_\t_\n"
        "3\ta\tab\tADP\tADP\t_\t4\tcase\t_\t_\n"
        "4\tCicerone\tCicero\tPROPN\tPROPN\tCase=Abl|Gender=Masc|Number=Sing\t1\tobl\t_\tSpaceAfter=No\n"
        "5\t.\t.\tPUNCT\tPUNCT\t_\t1\tpunct\t_\t_\n"
        "\n"
    )
    result_conllu, changes, report = harmonize_conllu(sample)

    # Check that the change was made
    assert len(changes) > 0
    obl_agent_changes = [c for c in changes if c.new_value == "obl:agent"]
    assert len(obl_agent_changes) == 1
    assert obl_agent_changes[0].old_value == "obl"
    assert "obl:agent" in result_conllu


# ---------------------------------------------------------------------------
# Integration tests: Flask endpoints
# ---------------------------------------------------------------------------

def test_analyze_returns_postprocessor_fields(client):
    """POST /api/analyze should return change_count and changes fields."""
    response = client.post(
        "/api/analyze",
        data=json.dumps({"text": "Gallia est omnis divisa in partes tres."}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "html" in data
    assert "tokens" in data
    assert "conllu" in data
    assert "raw_conllu" in data
    assert "change_count" in data
    assert isinstance(data["change_count"], int)
    assert "changes" in data
    assert isinstance(data["changes"], list)


def test_dev_compare_returns_both_parses(client):
    """POST /dev/compare should return raw and harmonized SVGs."""
    response = client.post(
        "/dev/compare",
        data=json.dumps({"text": "Roma pulchra est."}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "raw_html" in data
    assert "harmonized_html" in data
    assert "raw_conllu" in data
    assert "harmonized_conllu" in data
    assert "change_count" in data
    assert "changes" in data
    assert len(data["raw_html"]) > 0
    assert len(data["harmonized_html"]) > 0


def test_dev_home_returns_200(client):
    """GET /dev/ should return 200."""
    response = client.get("/dev/")
    assert response.status_code == 200
    assert b"Raw vs. Harmonized" in response.data
