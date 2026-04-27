"""Lexicons used by the rule modules.

The verb frames and adjective lists below are grounded in the UDante
Epistole dev set: every (head_lemma, preposition, case) triple with >=1
obl:arg occurrence observed is represented. Canonical seeds from Allen
& Greenough and standard Latin reference grammars are merged in.

Expand these lists as error analysis reveals more cases. Each list is
sorted for easy visual diff in version control.
"""

# ---------------------------------------------------------------------------
# T1.1 Rule B: verbs that take a DATIVE argument (no preposition)
# that UD labels obl:arg.
# ---------------------------------------------------------------------------
DATIVE_ARG_VERBS: set = {
    # --- observed in UDante Epistole dev ---
    "adscribo",
    "aggrego",
    "ancillor",
    "annuo",
    "appareo",
    "ascribo",
    "assurgo",
    "coniugo",
    "commendo",
    "confero",
    "confido",
    "congruo",
    "contendo",
    "debeo",
    "derogo",
    "destino",
    "desum",
    "distribuo",
    "do",
    "dominor",
    "exhibeo",
    "expedio",
    "fero",
    "ignosco",
    "indigeo",
    "indulgeo",
    "insidio",
    "insidior",
    "instituo",
    "intersum",
    "invideo",
    "mando",
    "ministro",
    "misereor",
    "noceo",
    "obicio",
    "oboedio",
    "obvio",
    "offero",
    "ostendo",
    "pareo",
    "persolvo",
    "provideo",
    "recalcitro",
    "recommendo",
    "reddo",
    "refero",
    "relinquo",
    "reluctor",
    "repugno",
    "resisto",
    "respondeo",
    "scribo",
    "significo",
    "subdo",
    "subicio",
    "succedo",
    "traduco",
    # --- classical dative-verb seeds not yet observed ---
    "benedico",
    "careo",
    "cedo",
    "credo",
    "diffido",
    "displiceo",
    "egeo",
    "faveo",
    "fido",
    "impero",
    "irascor",
    "maledico",
    "medeor",
    "minor",
    "nubo",
    "occurro",
    "parco",
    "persuadeo",
    "placeo",
    "servio",
    "studeo",
    "submitto",
    "subvenio",
    "succurro",
    "suadeo",
    "supplico",
}


# ---------------------------------------------------------------------------
# T1.1 Rule A: adjectives that govern a case argument labeled obl:arg.
# Default is dative; many also take ablative, and a subset take genitive.
# ---------------------------------------------------------------------------
CASE_GOVERNING_ADJECTIVES: set = {
    # --- observed in UDante Epistole dev ---
    "alienus",
    "coaequalis",
    "conformis",
    "congruus",
    "contrarius",
    "decorus",
    "dignus",
    "exsors",
    "immunis",
    "manifestus",
    "obnoxius",
    "plenus",
    "proximus",
    "securus",
    # --- classical case-governing adjectives (Allen & Greenough §§384-385) ---
    "acceptus",
    "aequalis",
    "amicus",
    "aptus",
    "avidus",
    "carus",
    "conscius",
    "consors",
    "cupidus",
    "dissimilis",
    "expers",
    "fidelis",
    "gratus",
    "idoneus",
    "ignarus",
    "immemor",
    "inanis",
    "indignus",
    "inimicus",
    "infidelis",
    "inops",
    "inscius",
    "inutilis",
    "invisus",
    "ingratus",
    "liber",
    "memor",
    "necessarius",
    "orbus",
    "par",
    "particeps",
    "peritus",
    "imperitus",
    "propinquus",
    "rudis",
    "similis",
    "studiosus",
    "utilis",
    "vacuus",
    "vicinus",
}


# Adjectives whose obl:arg dependent is in the Genitive (observed + canonical).
CASE_ADJ_GEN: set = {
    # observed
    "contrarius",
    "exsors",
    "immunis",
    # canonical
    "avidus",
    "conscius",
    "consors",
    "cupidus",
    "expers",
    "ignarus",
    "immemor",
    "imperitus",
    "inops",
    "memor",
    "particeps",
    "peritus",
    "plenus",  # plenus also takes Gen
    "studiosus",
}


# ---------------------------------------------------------------------------
# T1.1 Rule C: full (preposition, case, verb-lemma) frames.
# Dispatch key format: "{prep_lowercase}_{case_short}"
#   e.g. "ab_abl", "ad_acc", "in_abl", "de_abl"
# ---------------------------------------------------------------------------
OBL_ARG_VERB_FRAMES: dict = {
    "ab_abl": {
        # observed in UDante Epistole
        "abstineo",
        "absum",
        "cesso",
        "dependeo",
        "differo",
        "elongo",
        "segrego",
        "separo",
        # canonical seeds
        "abduco",
        "desisto",
        "discedo",
        "recedo",
        "removeo",
    },
    "ad_acc": {
        # observed
        "accedo",
        "attendo",
        "cogo",
        "consecro",
        "loquor",
        "perduco",
        "proximus",  # adjective, included for completeness
        "recurro",
        "remeo",
        "respicio",
        "revoco",
        "torqueo",
        "traduco",
        "verto",
        # canonical seeds
        "adhortor",
        "adigo",
        "aspiro",
        "attineo",
        "compello",
        "confugio",
        "exhortor",
        "hortor",
        "intendo",
        "pertineo",
        "redeo",
        "tendo",
    },
    "in_acc": {
        # observed
        "ardeo",
        "cado",
        "consentio",
        "corruo",
        "exardesco",
        "irrumpo",
        "reduco",
        "regno",
        "repatrio",
        "subigo",
        "trado",
        "verto",
        # canonical seeds
        "impingo",
        "incido",
        "irruo",
        "prorumpo",
    },
    "in_abl": {
        # observed
        "consisto",
        "credo",
        "exulto",
        "verto",
        # canonical seeds
        "glorior",
        "versor",
        "vigilo",
    },
    "de_abl": {
        # observed
        "concipio",
        "confido",
        "dubito",
        "excuso",
        "mitto",
        "prorumpo",
        "provideo",
        "removeo",
        # canonical seeds
        "dico",
        "disputo",
        "loquor",
        "scribo",
        "tracto",
    },
    "pro_abl": {
        # observed
        "propugno",
        # canonical seeds
        "certo",
        "deprecor",
        "dimico",
        "oro",
        "pugno",
    },
    "cum_abl": {
        # observed
        "consentio",
        # canonical seeds
        "certo",
        "colloquor",
        "communico",
        "congredior",
        "contendo",
        "pugno",
    },
}


# ---------------------------------------------------------------------------
# T1.1 Rule D: lemmas that take a bare ablative argument labeled obl:arg.
# ---------------------------------------------------------------------------
BARE_ABL_ARG_LEMMAS: set = {
    # --- observed in UDante Epistole dev ---
    "armo",
    "abutor",
    "ardo",
    "cesso",
    "decorus",       # adjective
    "destituo",
    "gaudeo",
    "indigeo",
    "infigo",
    "iuvo",
    "mano",
    "metior",
    "molior",
    "plenus",        # adjective
    "subeo",
    "vaco",          # classical spelling
    "uaco",          # UDante-style spelling
    "vigilo",
    # --- canonical seeds: utor family, deprivation, means ---
    "careo",
    "egeo",
    "exuo",
    "fruor",
    "fungor",
    "potior",
    "utor",
    "vescor",
}


# ---------------------------------------------------------------------------
# T1.3: obl -> obl:agent
# ---------------------------------------------------------------------------

# Deponents that behave like passives for agent marking.
# Used when the ablative is governed by "a/ab" or is otherwise an agent.
DEPONENT_LEMMAS: set = {
    "admiror",
    "adhortor",
    "amplexor",
    "comitor",
    "confiteor",
    "conor",
    "deprecor",
    "exhortor",
    "fabulor",
    "fruor",
    "fungor",
    "glorior",
    "hortor",
    "loquor",
    "mentior",
    "miror",
    "misereor",
    "morior",
    "nascor",
    "obliviscor",
    "orior",
    "patior",
    "polliceor",
    "potior",
    "precor",
    "queror",
    "reor",
    "sequor",
    "suspicor",
    "tueor",
    "utor",
    "vereor",
    "versor",
}


# Prepositions that mark an agent when the head is passive or deponent.
AGENT_PREPOSITIONS: set = {"a", "ab", "abs"}


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
PASSIVE_VOICE_VALUES = {"Pass"}