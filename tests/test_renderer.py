"""Tests for Phase 4: Sentence Map Renderer."""
import xml.etree.ElementTree as ET
from latin_reader.pipeline.chunker import chunk_sentence
from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string
from latin_reader.pipeline.renderer import render_sentence_map

def test_render_valid_svg():
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
    sentences = list(read_conllu_string(conllu))
    tokens = sentences[0].tokens
    chunk = chunk_sentence(tokens)
    
    svg_str = render_sentence_map(chunk, tokens)
    
    assert svg_str.startswith("<svg")
    assert svg_str.strip().endswith("</svg>")
    assert "Galliam" in svg_str
    assert "Cum" in svg_str
    
    # Assert valid XML
    try:
        ET.fromstring(svg_str)
    except ET.ParseError as e:
        assert False, f"SVG is not valid XML: {e}"

def test_render_graceful_degradation():
    """Ensure that if the chunks are completely broken, it yields some safe HTML."""
    from latin_reader.pipeline.renderer import render_fallback_html
    from latin_reader.latincy_postprocessor.sentence import Token
    from latin_reader.pipeline.chunker import Chunk
    
    # Minimal flat sentence
    t = Token(1, "bad", "bad", "NOUN", "NOUN", head=0, deprel="root")
    c = Chunk(type="Sentence", label="Bad", token_ids=[1])
    
    html = render_fallback_html(c, [t])
    assert html.startswith("<ul")
    assert "bad" in html

def test_api_render_endpoint(client):
    conllu = (
        "1\tRoma\tRoma\tPROPN\tPROPN\t_\t3\tnsubj\t_\t_\n"
        "2\test\tsum\tAUX\tAUX\t_\t0\troot\t_\t_\n\n"
    )
    response = client.post(
        "/api/render",
        json={"conllu": conllu}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "svg" in data
    assert data["svg"].startswith("<svg")
