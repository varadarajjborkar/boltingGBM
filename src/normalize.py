"""Text normalisation for business names and addresses.

Everything here is deterministic, uses only the provided data plus general language
knowledge (abbreviations, legal forms, state names), and is country-agnostic so an
unseen country (France in test) goes through the same code path.
"""
import re
import unicodedata
from functools import lru_cache

from anyascii import anyascii

# ---------------------------------------------------------------- shared helpers
_JUNK_EDGE = re.compile(r"^[\s>\-<#*~=|:._]+|[\s>\-<#*~=|:._]+$")
_ID_TAG = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\s?\d{3,}\b", re.I)          # "(ID: 33664)", "#30780"
_ALT_SPLIT = re.compile(r"\s*(?:\bf/?k/?a\b|\bd/?b/?a\b|\ba/?k/?a\b|\bformerly\b|\btrading as\b|\bt/a\b|\|)\s*", re.I)
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]{2,})\.(?:com|net|org|co\.in|in|co|biz|fr|io|info|us)/?$", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_HAS_ALPHA = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"[0-9]")


_NUKTA = {cp: None for cp in (0x093C, 0x09BC, 0x0A3C, 0x0ABC, 0x0B3C, 0x0CBC)}   # nukta signs


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _to_ascii(s: str) -> str:
    # Tamil uses the aytham + pa ("ஃப") for the English f sound; anyascii would emit "kp".
    s = s.replace("ஃப", "f").replace("ஃஜ", "z")
    s = s.translate(_NUKTA)
    # Malayalam double-retroflex "റ്റ" (as in private/limited) is "tt", anyascii gives "rr"; "ന്റ" is "nd"
    s = s.replace("\u0d31\u0d4d\u0d31", "t").replace("\u0d28\u0d4d\u0d31", "nd")
    return anyascii(s)


def _deleet_token(t: str) -> str:
    # only tokens that mix letters and digits, e.g. imp0rts, a1liance; keeps 3m / 24x7 style short codes
    nd = sum(ch.isdigit() for ch in t)
    if len(t) >= 4 and 1 <= nd <= 2 and len(t) - nd >= 3:
        return t.translate(_LEET)
    return t


def basic_clean(s: str) -> str:
    """NFKC -> ascii (transliterates any script, folds accents) -> lower -> & to and -> alnum tokens."""
    s = _to_ascii(_nfkc(s)).lower().replace("&", " and ").replace("+", " and ")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


# ---------------------------------------------------------------- names
# canonical legal forms (general knowledge; applied to any country)
LEGAL = {
    "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lmited": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "snc": "snc", "scop": "scop", "selarl": "selarl",
    "cie": "co", "societe": "co",
}
# ambiguous short forms: only legal as the FIRST or LAST token ("P C Jeweller", "AG Traders", "Sa Re Ga" keep them)
LEGAL_LAST = {"pc": "pc", "pa": "pa", "sa": "sa", "ag": "ag", "sca": "sca"}      # "Siemens AG", "Dupont SA"
LEGAL_FIRST = {"sci": "sci", "ets": "co", "ste": "co"}                          # "SCI Ptit Amicale", "Ets Martin"
# multi-token forms, matched on the cleaned string before tokenising
_LEGAL_PHRASES = [   # spelled-out forms, only at the start or end of the name
    (re.compile(r"(^|\s)l l c($|\s)"), " llc "), (re.compile(r"(^|\s)l l p($|\s)"), " llp "),
    (re.compile(r"(^|\s)s a s( u)?$"), " sas "), (re.compile(r"(^|\s)s a r l($|\s)"), " sarl "),
    (re.compile(r"\sp c$"), " pc "), (re.compile(r"\bpte ltd\b"), " pvt ltd "),
]


_NAME_TOK = {"et": "and", "und": "and", "y": "and"}   # "Dupont et Fils" == "Dupont & Fils"


@lru_cache(maxsize=None)
def _norm_name_part(raw: str):
    non_latin = _is_non_latin(raw)
    s = _ID_TAG.sub(" ", raw)
    s = _JUNK_EDGE.sub("", s.strip())
    m = _DOMAIN.match(s.strip().lower())
    was_domain = bool(m)
    if m:
        s = m.group(1).replace("-", " ")
    if s.startswith("#"):
        s = s[1:]
    s = basic_clean(s)
    for pat, rep in _LEGAL_PHRASES:
        s = pat.sub(rep, s)
    toks = [_NAME_TOK.get(t, t) for t in (_deleet_token(t) for t in s.split() if not (t.isdigit() and len(t) >= 7))]
    core, legal = [], []
    last = len(toks) - 1
    # transliterated legal words are stripped only from the END (so "Parvati Stores" keeps "Parvati")
    tail = last + 1
    if non_latin:
        while tail > 0 and len(phonetic_key(toks[tail - 1])) >= 3 and phonetic_key(toks[tail - 1]) in _LEGAL_PHON:
            tail -= 1
    for k, t in enumerate(toks):
        if k >= tail:
            legal.append(_LEGAL_PHON[phonetic_key(t)])
        elif t in LEGAL:
            legal.append(LEGAL[t])
        elif t in LEGAL_LAST and k == last and len(toks) > 1:
            legal.append(LEGAL_LAST[t])
        elif t in LEGAL_FIRST and k == 0 and len(toks) > 1:
            legal.append(LEGAL_FIRST[t])
        elif t == "public" and k < last and toks[k + 1] in ("ltd", "limited", "co", "company"):
            legal.append("public")
        else:
            core.append(t)
    while core and core[-1] in ("and", "the", "of"):   # "dupont et cie" -> "dupont"
        core.pop()
    if not core and legal:  # name made only of legal words: keep them as the core
        core = legal[:]
    return " ".join(core), " ".join(sorted(set(legal))), was_domain


def _is_non_latin(raw: str) -> bool:
    return any(ord(c) > 0x024F and not unicodedata.category(c).startswith("P") for c in raw if not c.isspace())


def normalize_name(raw: str):
    """Returns dict: core (main key), legal (sorted legal forms), alts (other f/k/a names), flags."""
    raw = _nfkc(raw)
    parts = [p for p in _ALT_SPLIT.split(raw) if p and p.strip()]
    if not parts:
        parts = [raw]
    main = _norm_name_part(parts[0])
    alts = [_norm_name_part(p)[0] for p in parts[1:]]
    alts = [a for a in alts if a and a != main[0]]
    return {
        "core": main[0],
        "legal": main[1],
        "alts": "|".join(alts),
        "is_domain": main[2],
        "has_alt": bool(alts),
        "non_latin": _is_non_latin(raw),
    }


_STOP_CAT = {"and", "the", "of"}


def concat_key(core: str) -> str:
    """'balash and arredondo' -> 'balasharredondo' (matches domain-style names)."""
    return "".join(t for t in core.split() if t not in _STOP_CAT)


def acronym_key(core: str) -> str:
    """'ace foundation' -> 'af'; empty for single-token names."""
    toks = [t for t in core.split() if t not in _STOP_CAT]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


# ---------------------------------------------------------------- phonetic key (for transliterated names)
_PH_RULES = [
    ("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("sh", "s"),
    ("ch", "c"), ("ck", "k"), ("qu", "k"), ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v"), ("y", "i"),
    # voiced -> unvoiced (Tamil script has no voicing distinction)
    ("d", "t"), ("g", "k"), ("b", "p"), ("c", "k"),
]
_ANUSVARA = re.compile(r"(?<=[a-z])m(?=[^aeioum\s])")   # non-initial m before a consonant -> n (anusvara)
_VOWELS_NOT_FIRST = re.compile(r"(?<!^)(?<! )[aeiou]")
_REPEAT = re.compile(r"(.)\1+")


@lru_cache(maxsize=None)
def phonetic_key(core: str) -> str:
    s = _ANUSVARA.sub("n", core)
    for a, b in _PH_RULES:
        s = s.replace(a, b)
    s = _VOWELS_NOT_FIRST.sub("", s)
    s = _REPEAT.sub(r"\1", s)
    return _SPACES.sub(" ", s).strip()


_LEGAL_PHON = {}


def _build_legal_phon():
    for w, canon in [("private", "pvt"), ("limited", "ltd"), ("company", "co"), ("corporation", "corp"),
                     ("incorporated", "inc"), ("public", "public"), ("pvt", "pvt"), ("ltd", "ltd")]:
        _LEGAL_PHON[phonetic_key(w)] = canon


# ---------------------------------------------------------------- addresses
# canonical short forms (US, India, France); both sides map to the same token
ADDR_ABBR = {
    # US / generic street types
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd", "lane": "ln", "court": "ct", "cove": "cv", "place": "pl",
    "terrace": "ter", "circle": "cir", "highway": "hwy", "parkway": "pkwy", "trail": "trl", "square": "sq",
    "point": "pt", "mount": "mt", "fort": "ft", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "apartment": "apt", "suite": "ste",
    "floor": "fl", "building": "bldg", "unit": "unit", "route": "rte", "expressway": "expy", "freeway": "fwy",
    "crossing": "xing", "heights": "hts", "junction": "jct", "center": "ctr", "centre": "ctr", "plaza": "plz",
    "ridge": "rdg", "village": "vlg", "township": "twp", "county": "cnty",
    # India
    "near": "nr", "opposite": "opp", "number": "no", "nagar": "ngr", "colony": "colony", "sector": "sec",
    "phase": "ph", "district": "dist", "taluka": "tal", "tq": "tal", "post": "po", "marg": "marg",
    "house": "h", "hno": "h", "flat": "flat", "plot": "plot", "door": "door", "dno": "door", "flno": "flat",
    "ground": "gr", "first": "1st", "second": "2nd", "third": "3rd",
    # France
    "rue": "rue", "r": "rue", "allee": "all", "impasse": "imp", "chemin": "ch", "chem": "ch", "lotissement": "lot",
    "residence": "res", "faubourg": "fbg", "quai": "qu", "cours": "crs", "passage": "pass", "promenade": "prom",
    "sainte": "ste",
}
_DROP_ADDR = {"null", "none", "na", "nan", "the", "of", "c", "o"}   # "c/o" becomes "c o"

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli and daman and diu": "dn",
    # native-language names that are not sound-alikes of the English name
    "paschimbanga": "wb", "pashchimbanga": "wb", "paschim banga": "wb", "pscimbng": "wb", "bangla": "wb",
    "dilli": "dl", "nct of delhi": "dl", "odisa": "od",
}
IN_CODES = {"ap", "ar", "as", "br", "cg", "ct", "ga", "gj", "hr", "hp", "jh", "ka", "kl", "mp", "mh", "mn", "ml",
            "mz", "nl", "od", "or", "pb", "rj", "sk", "tn", "ts", "tg", "tr", "up", "uk", "ut", "wb", "dl", "jk",
            "ch", "py", "an", "ld", "dn"}
_IN_CODE_ALIAS = {"ct": "cg", "or": "od", "tg": "ts", "ut": "uk"}
_US_NAME2CODE = {v: k for k, v in US_STATES.items()}
_DIGITS = re.compile(r"\d+")
_NUM_TOKEN = re.compile(r"\d+")



# France: regions (post-2016 + common pre-2016 names) and departements -> region code. General administrative
# knowledge (like US state names), used so that France gets a "state" the same way US/India do.
FR_REGION_DEPTS = {
    "ara": ["auvergne rhone alpes", "rhone alpes", "auvergne", "ain", "allier", "ardeche", "cantal", "drome", "isere",
            "loire", "haute loire", "puy de dome", "rhone", "savoie", "haute savoie", "metropole de lyon"],
    "bfc": ["bourgogne franche comte", "bourgogne", "franche comte", "cote d or", "doubs", "jura", "nievre",
            "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "bre": ["bretagne", "cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "alsace", "lorraine", "champagne ardenne", "ardennes", "aube", "marne", "haute marne",
            "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin", "vosges"],
    "hdf": ["hauts de france", "nord pas de calais", "picardie", "aisne", "nord", "oise", "pas de calais", "somme"],
    "idf": ["ile de france", "paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
            "val de marne", "val d oise"],
    "nor": ["normandie", "haute normandie", "basse normandie", "calvados", "eure", "manche", "orne", "seine maritime"],
    "naq": ["nouvelle aquitaine", "aquitaine", "poitou charentes", "limousin", "charente", "charente maritime",
            "correze", "creuse", "dordogne", "gironde", "landes", "lot et garonne", "pyrenees atlantiques",
            "deux sevres", "vienne", "haute vienne"],
    "occ": ["occitanie", "midi pyrenees", "languedoc roussillon", "ariege", "aude", "aveyron", "gard",
            "haute garonne", "gers", "herault", "lot", "lozere", "hautes pyrenees", "pyrenees orientales", "tarn",
            "tarn et garonne"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "pac": ["provence alpes cote d azur", "paca", "alpes de haute provence", "hautes alpes", "alpes maritimes",
            "bouches du rhone", "var", "vaucluse"],
}
FR_STATES = {n: code for code, names in FR_REGION_DEPTS.items() for n in names}

_IN_STATE_PHON = None


def _in_state_phon():
    global _IN_STATE_PHON
    if _IN_STATE_PHON is None:
        _IN_STATE_PHON = {k: c for n, c in IN_STATES.items() if len(k := phonetic_key(n).replace(" ", "")) >= 3}
    return _IN_STATE_PHON


def _find_state(raw: str, country_hint: str):
    """State from whole comma-separated components (last match wins). Returns canonical code or ''.

    Works on components so that e.g. "CT" meaning Court inside a street is not taken for Connecticut.
    Native-script Indian state names are matched through the phonetic key.
    """
    found = ""
    for comp in (raw or "").split(","):
        c = basic_clean(comp)
        if not c:
            continue
        if country_hint != "India":
            if c in _US_NAME2CODE:
                found = _US_NAME2CODE[c]
                continue
            if country_hint == "US" and c in US_STATES:
                found = c
                continue
        if country_hint != "US":
            if c in IN_STATES:
                found = IN_STATES[c]
                continue
            if country_hint == "India" and c in IN_CODES:
                found = _IN_CODE_ALIAS.get(c, c)
                continue
            if country_hint == "India" and any(ord(ch) > 0x0900 for ch in comp):
                k = phonetic_key(c).replace(" ", "")
                code = _in_state_phon().get(k)
                if code:
                    found = code
    return found


def _state_of_component(c: str, comp_raw: str, country_hint: str) -> str:
    c = _DIGITS.sub(" ", c).strip() if _HAS_DIGIT.search(c) else c
    c = _SPACES.sub(" ", c)
    if not c:
        return ""
    if country_hint not in ("US", "India") and c in FR_STATES:   # France (and any other country: try it too)
        return FR_STATES[c]
    if country_hint != "India":
        if c in _US_NAME2CODE:
            return _US_NAME2CODE[c]
        if country_hint == "US" and c in US_STATES:
            return c
    if country_hint != "US":
        if c in IN_STATES:
            return IN_STATES[c]
        if country_hint == "India" and c in IN_CODES:
            return _IN_CODE_ALIAS.get(c, c)
        if country_hint == "India" and any(ord(ch) > 0x0900 for ch in comp_raw):
            k = phonetic_key(c).replace(" ", "")
            return _in_state_phon().get(k, "") if len(k) >= 3 else ""
    return ""


@lru_cache(maxsize=None)
def normalize_address(raw: str, country: str):
    """Returns dict: text (canonical tokens, state as its code), state, nums (numbers w/o leading zeros), flags.

    Works per comma-separated component so a state is only recognised when it is a whole component.
    """
    raw = _nfkc(raw)
    out, state, n_tok = [], "", 0
    for comp in raw.split(","):
        c = basic_clean(comp)
        if not c:
            continue
        code = _state_of_component(c, comp, country)
        if code:
            state = code
            out.append(code)
            n_tok += 1
            continue
        toks = [t for t in c.split() if t not in _DROP_ADDR]
        n_tok += len(toks)
        out.extend(ADDR_ABBR.get(t, t) for t in toks)
    nums = sorted({str(int(n)) for t in out for n in _NUM_TOKEN.findall(t) if len(n) <= 7})
    return {
        "text": " ".join(out),
        "state": state,
        "nums": " ".join(nums),
        "empty": n_tok == 0,
        "native_state": any(ord(c) > 0x0900 for c in raw),
    }


_build_legal_phon()
