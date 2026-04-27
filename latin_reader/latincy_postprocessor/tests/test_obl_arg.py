from latincy_postprocessor.sentence import Sentence, Token
from latincy_postprocessor.rules.obl_arg import OblArgRule


def _make_token(id, form, lemma, upos, feats, head, deprel):
    return Token(id=id, form=form, lemma=lemma, upos=upos, xpos="_",
                 feats=feats, head=head, deprel=deprel)


def test_cessare_ab():
    # "ab ... insultu cessaremus" — Epistole 11
    s = Sentence()
    s.comments = ["# sent_id = test-cesso-ab"]
    s.tokens = [
        _make_token(1, "ab", "ab", "ADP", {}, 2, "case"),
        _make_token(2, "insultu", "insultus", "NOUN",
                    {"Case": "Abl", "Number": "Sing"},
                    3, "obl"),
        _make_token(3, "cessaremus", "cesso", "VERB",
                    {"VerbForm": "Fin"}, 0, "root"),
    ]
    changes = OblArgRule().apply(s)
    assert len(changes) == 1
    assert s.by_id(2).deprel == "obl:arg"


def test_intendere_ad():
    # "ad sulcos ... intenditis" — Epistole 8
    s = Sentence()
    s.comments = ["# sent_id = test-intendere-ad"]
    s.tokens = [
        _make_token(1, "ad", "ad", "ADP", {}, 2, "case"),
        _make_token(2, "sulcos", "sulcus", "NOUN",
                    {"Case": "Acc", "Number": "Plur"},
                    3, "obl"),
        _make_token(3, "intenditis", "intendo", "VERB",
                    {"VerbForm": "Fin"}, 0, "root"),
    ]
    changes = OblArgRule().apply(s)
    assert len(changes) == 1


def test_unknown_verb_not_promoted():
    # Generic locative oblique with no frame entry stays obl.
    s = Sentence()
    s.comments = ["# sent_id = test-unknown"]
    s.tokens = [
        _make_token(1, "in", "in", "ADP", {}, 2, "case"),
        _make_token(2, "horto", "hortus", "NOUN",
                    {"Case": "Abl", "Number": "Sing"},
                    3, "obl"),
        _make_token(3, "ambulat", "ambulo", "VERB",
                    {"VerbForm": "Fin"}, 0, "root"),
    ]
    changes = OblArgRule().apply(s)
    assert changes == []


def test_dative_indigere():
    s = Sentence()
    s.comments = ["# sent_id = test-indigere"]
    s.tokens = [
        _make_token(1, "consiliis", "consilium", "NOUN",
                    {"Case": "Dat", "Number": "Plur"},
                    2, "obl"),
        _make_token(2, "indigeat", "indigeo", "VERB",
                    {"VerbForm": "Fin"}, 0, "root"),
    ]
    changes = OblArgRule().apply(s)
    assert len(changes) == 1
    assert s.by_id(1).deprel == "obl:arg"