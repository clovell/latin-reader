"""Tests for T1.2 using real examples from UDante Epistole 2, 7, 11."""
import pytest
from latincy_postprocessor.sentence import Sentence, Token
from latincy_postprocessor.rules.advcl_pred import AdvclPredRule


def _make_token(id, form, lemma, upos, feats, head, deprel):
    return Token(id=id, form=form, lemma=lemma, upos=upos, xpos="_",
                 feats=feats, head=head, deprel=deprel)


def test_moniti_promoted():
    # Epistole 2: "Preceptis salutaribus moniti ... respondemus."
    # token 3 'moniti' (participle) agrees with implicit subject of
    # 'respondemus' — but more concretely, it has sibling at same head
    # token 20 and should match against nsubj. For this test we build
    # a minimal agreeing structure: 'moniti' advcl on 'respondemus';
    # no overt nsubj, but we give respondemus a fake nsubj that agrees.
    s = Sentence()
    s.comments = ["# sent_id = test-moniti"]
    s.tokens = [
        _make_token(1, "nos", "ego", "PRON",
                    {"Case": "Nom", "Number": "Plur", "Gender": "Masc"},
                    2, "nsubj"),
        _make_token(2, "respondemus", "respondeo", "VERB",
                    {"VerbForm": "Fin", "Voice": "Act"},
                    0, "root"),
        _make_token(3, "moniti", "moneo", "VERB",
                    {"VerbForm": "Part", "Case": "Nom",
                     "Number": "Plur", "Gender": "Masc", "Voice": "Pass"},
                    2, "advcl"),
    ]
    changes = AdvclPredRule().apply(s)
    assert len(changes) == 1
    assert changes[0].token_id == 3
    assert s.by_id(3).deprel == "advcl:pred"


def test_no_agreement_no_change():
    s = Sentence()
    s.comments = ["# sent_id = test-no-agreement"]
    s.tokens = [
        _make_token(1, "puella", "puella", "NOUN",
                    {"Case": "Nom", "Number": "Sing", "Gender": "Fem"},
                    2, "nsubj"),
        _make_token(2, "venit", "venio", "VERB",
                    {"VerbForm": "Fin"},
                    0, "root"),
        _make_token(3, "moniti", "moneo", "VERB",
                    {"VerbForm": "Part", "Case": "Nom",
                     "Number": "Plur", "Gender": "Masc"},
                    2, "advcl"),
    ]
    changes = AdvclPredRule().apply(s)
    assert changes == []
    assert s.by_id(3).deprel == "advcl"


def test_non_participle_untouched():
    s = Sentence()
    s.comments = ["# sent_id = test-non-part"]
    s.tokens = [
        _make_token(1, "puer", "puer", "NOUN",
                    {"Case": "Nom", "Number": "Sing", "Gender": "Masc"},
                    2, "nsubj"),
        _make_token(2, "discit", "disco", "VERB",
                    {"VerbForm": "Fin"}, 0, "root"),
        _make_token(3, "legit", "lego", "VERB",
                    {"VerbForm": "Fin"}, 2, "advcl"),
    ]
    changes = AdvclPredRule().apply(s)
    assert changes == []