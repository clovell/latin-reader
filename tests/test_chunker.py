"""Tests for Phase 3: Chunker."""
import pytest
import json

from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
from latin_reader.pipeline.chunker import chunk_sentence, Chunk


def _get_sentence_chunk(conllu_data: str) -> Chunk:
    sentences = list(read_conllu_string(conllu_data))
    return chunk_sentence(sentences[0].tokens)


def _find_chunk(chunk: Chunk, type_name: str) -> Chunk:
    """Helper to find the first chunk of a given type."""
    if chunk.type == type_name:
        return chunk
    for child in chunk.children:
        found = _find_chunk(child, type_name)
        if found is not None:
            return found
    return None


def test_chunker_simple_sentence():
    """Test basic Subject + Verb chunking."""
    conllu = (
        "1\tCaesar\tCaesar\tPROPN\tPROPN\t_\t2\tnsubj\t_\t_\n"
        "2\tvicit\tvinco\tVERB\tVERB\t_\t0\troot\t_\t_\n"
        "3\t.\t.\tPUNCT\tPUNCT\t_\t2\tpunct\t_\t_\n"
        "\n"
    )
    chunk = _get_sentence_chunk(conllu)
    assert chunk.type == "Sentence"
    
    main_clause = _find_chunk(chunk, "MainClause")
    assert main_clause is not None
    assert main_clause.head_token_id == 2
    
    np_chunk = _find_chunk(main_clause, "NP")
    assert np_chunk is not None
    assert np_chunk.head_token_id == 1
    assert np_chunk.token_ids == [1]

def test_chunker_pp():
    """Test grouping of a prepositional phrase."""
    conllu = (
        "1\tin\tin\tADP\tADP\t_\t2\tcase\t_\t_\n"
        "2\tGalliam\tGallia\tPROPN\tPROPN\t_\t3\tobl\t_\t_\n"
        "3\tvenit\tvenio\tVERB\tVERB\t_\t0\troot\t_\t_\n\n"
    )
    chunk = _get_sentence_chunk(conllu)
    pp_chunk = _find_chunk(chunk, "PP")
    assert pp_chunk is not None
    assert pp_chunk.token_ids == [1, 2]

def test_chunker_ablative_absolute():
    """Test Ablative Absolute identification."""
    conllu = (
        "1\tCaesare\tCaesar\tPROPN\tPROPN\t_\t2\tnsubj\t_\t_\n"
        "2\tduce\tdux\tNOUN\tNOUN\t_\t4\tadvcl:abs\t_\t_\n"
        "3\thostes\thostis\tNOUN\tNOUN\t_\t4\tnsubj\t_\t_\n"
        "4\tfugerunt\tfugio\tVERB\tVERB\t_\t0\troot\t_\t_\n\n"
    )
    chunk = _get_sentence_chunk(conllu)
    ab_abs = _find_chunk(chunk, "AblativeAbsolute")
    assert ab_abs is not None
    assert ab_abs.head_token_id == 2
    # Caesare is dependent on duce
    assert 1 in ab_abs.flatten_token_ids()

def test_chunker_acceptance_sentence():
    """Test 'Cum Caesar in Galliam venisset, hostes fugerunt'."""
    conllu = (
        "1\tCum\tcum\tSCONJ\tSCONJ\t_\t5\tmark\t_\t_\n"
        "2\tCaesar\tCaesar\tPROPN\tPROPN\t_\t5\tnsubj\t_\t_\n"
        "3\tin\tin\tADP\tADP\t_\t4\tcase\t_\t_\n"
        "4\tGalliam\tGallia\tPROPN\tPROPN\t_\t5\tobl\t_\t_\n"
        "5\tvenisset\tvenio\tVERB\tVERB\t_\t8\tadvcl\t_\tSpaceAfter=No\n"
        "6\t,\t,\tPUNCT\tPUNCT\t_\t5\tpunct\t_\t_\n"
        "7\thostes\thostis\tNOUN\tNOUN\t_\t8\tnsubj\t_\t_\n"
        "8\tfugerunt\tfugio\tVERB\tVERB\t_\t0\troot\t_\tSpaceAfter=No\n"
        "9\t.\t.\tPUNCT\tPUNCT\t_\t8\tpunct\t_\t_\n\n"
    )
    chunk = _get_sentence_chunk(conllu)
    
    sub = _find_chunk(chunk, "SubordinateClause")
    assert sub is not None
    assert sub.head_token_id == 5

    main = _find_chunk(chunk, "MainClause")
    assert main is not None
    assert main.head_token_id == 8

    # Inside SubordinateClause
    assert 1 in sub.token_ids  # The 'Cum' mark
    pp = _find_chunk(sub, "PP")
    assert pp is not None
    assert pp.token_ids == [3, 4]
    
    subj = _find_chunk(sub, "NP")
    assert subj is not None
    assert subj.head_token_id == 2

def test_chunker_fallback():
    """Test that a circular/bad dependency graph gracefully falls back."""
    from latin_reader.latincy_postprocessor.sentence import Token
    # Circular dependency: 1 head is 2, 2 head is 1
    t1 = Token(1, "bad", "bad", "NOUN", "NOUN", head=2, deprel="root")
    t2 = Token(2, "graph", "graph", "NOUN", "NOUN", head=1, deprel="root")
    
    # Actually, the logic looks at 'root'/etc. It shouldn't infinite loop 
    # but we will force an exception if we can, or just confirm it doesn't crash.
    chunk = chunk_sentence([t1, t2])
    assert chunk.type == "Sentence"

def test_api_chunk_endpoint(client):
    conllu = (
        "1\tRoma\tRoma\tPROPN\tPROPN\t_\t3\tnsubj\t_\t_\n"
        "2\tpulchra\tpulcher\tADJ\tADJ\t_\t1\tamod\t_\t_\n"
        "3\test\tsum\tAUX\tAUX\t_\t0\troot\t_\t_\n\n"
    )
    response = client.post(
        "/api/chunk",
        data=json.dumps({"conllu": conllu}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "chunks" in data
    assert len(data["chunks"]) == 1
    
    root_chunk = data["chunks"][0]
    assert root_chunk["type"] == "Sentence"
    assert len(root_chunk["children"]) > 0
    
    main_clause = root_chunk["children"][0]
    assert main_clause["type"] == "MainClause"
