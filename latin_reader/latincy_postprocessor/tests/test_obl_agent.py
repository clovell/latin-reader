from latincy_postprocessor.sentence import Sentence, Token
from latincy_postprocessor.rules.obl_agent import OblAgentRule


def _make_token(id, form, lemma, upos, feats, head, deprel):
    return Token(id=id, form=form, lemma=lemma, upos=upos, xpos="_",
                 feats=feats, head=head, deprel=deprel)


def test_virtutibus_agent():
    # "a Virtutibus honoratur" — Epistole 19
    s = Sentence()
    s.comments = ["# sent_id = test-virtutibus"]
    s.tokens = [
        _make_token(1, "a", "a", "ADP", {}, 2, "case"),
        _make_token(2, "Virtutibus", "virtus", "NOUN",
                    {"Case": "Abl", "Number": "Plur"},
                    3, "obl"),
        _make_token(3, "honoratur", "honoro", "VERB",
                    {"VerbForm": "Fin", "Voice": "Pass"},
                    0, "root"),
    ]
    changes = OblAgentRule().apply(s)
    assert len(changes) == 1
    assert s.by_id(2).deprel == "obl:agent"


def test_no_preposition_no_change():
    # Conservative default: bare ablative is not promoted.
    s = Sentence()
    s.comments = ["# sent_id = test-bare-abl"]
    s.tokens = [
        _make_token(1, "gladiis", "gladius", "NOUN",
                    {"Case": "Abl", "Number": "Plur"},
                    2, "obl"),
        _make_token(2, "exuti", "exuo", "VERB",
                    {"VerbForm": "Part", "Voice": "Pass"},
                    0, "root"),
    ]
    changes = OblAgentRule().apply(s)
    assert changes == []


def test_active_verb_no_change():
    s = Sentence()
    s.comments = ["# sent_id = test-active"]
    s.tokens = [
        _make_token(1, "ab", "ab", "ADP", {}, 2, "case"),
        _make_token(2, "urbe", "urbs", "NOUN",
                    {"Case": "Abl", "Number": "Sing"},
                    3, "obl"),
        _make_token(3, "venit", "venio", "VERB",
                    {"VerbForm": "Fin", "Voice": "Act"},
                    0, "root"),
    ]
    changes = OblAgentRule().apply(s)
    assert changes == []