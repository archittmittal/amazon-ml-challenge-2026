"""Record canonicalisation: script transliteration, de-obfuscation, tokenisation.

Everything here is rule-based and country-agnostic. Language-specific knowledge is
limited to small, generic abbreviation lists; everything else (state codes, local
abbreviations, transliteration residue) is learned later by `equivalence.py` from
the data itself, which is what lets the pipeline handle France with no labels.
"""
import re
import unicodedata

# --------------------------------------------------------------------------------------
# 1. Brahmic transliteration.
# The nine major Indic scripts in Unicode (Devanagari U+0900 ... Malayalam U+0D00) share
# one 128-codepoint layout inherited from ISCII: the same offset is the same phoneme in
# every script. A single offset table therefore transliterates all of them.
# --------------------------------------------------------------------------------------
_BRAHMIC_LO, _BRAHMIC_HI = 0x0900, 0x0D7F

_VOWEL = {0x05: 'a', 0x06: 'a', 0x07: 'i', 0x08: 'i', 0x09: 'u', 0x0A: 'u', 0x0B: 'ri',
          0x0C: 'li', 0x0D: 'e', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o',
          0x13: 'o', 0x14: 'au', 0x60: 'ri', 0x61: 'li'}
_CONS = {0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng', 0x1A: 'ch', 0x1B: 'chh',
         0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny', 0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh',
         0x23: 'n', 0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n', 0x29: 'n',
         0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm', 0x2F: 'y', 0x30: 'r',
         0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'zh', 0x35: 'v', 0x36: 'sh', 0x37: 'sh',
         0x38: 's', 0x39: 'h', 0x58: 'q', 0x59: 'kh', 0x5A: 'gh', 0x5B: 'z', 0x5C: 'r',
         0x5D: 'rh', 0x5E: 'f', 0x5F: 'y'}
_MATRA = {0x3E: 'a', 0x3F: 'i', 0x40: 'i', 0x41: 'u', 0x42: 'u', 0x43: 'ri', 0x44: 'ri',
          0x45: 'e', 0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o',
          0x4C: 'au', 0x57: 'au', 0x62: 'li', 0x63: 'li'}
_FINAL = {0x7A: 'n', 0x7B: 'n', 0x7C: 'r', 0x7D: 'l', 0x7E: 'l', 0x7F: 'k', 0x4E: 't'}  # chillu, khanda-ta
_NASAL = {0x01, 0x02, 0x70}           # candrabindu, anusvara, gurmukhi tippi
_VIRAMA = 0x4D
_NUKTA = 0x3C
_NUKTA_SHIFT = {'j': 'z', 'ph': 'f', 'k': 'q', 'd': 'r', 'dh': 'rh', 'g': 'g', 'kh': 'kh'}


def transliterate_brahmic(s):
    out = []
    pending = False          # last consonant still carries its inherent 'a'
    last_cons = None
    for ch in s:
        cp = ord(ch)
        if _BRAHMIC_LO <= cp <= _BRAHMIC_HI:
            off = cp & 0x7F
            if off == 0x71 and 0x0B00 <= cp < 0x0B80:   # Oriya wa
                off = 0x35
            if off in _CONS:
                if pending:
                    out.append('a')
                c = _CONS[off]
                out.append(c)
                last_cons = len(out) - 1
                pending = True
            elif off in _MATRA:
                out.append(_MATRA[off])
                pending = False
            elif off == _VIRAMA:
                pending = False
            elif off == _NUKTA:
                if last_cons is not None:
                    out[last_cons] = _NUKTA_SHIFT.get(out[last_cons], out[last_cons])
            elif off in _VOWEL:
                if pending:
                    out.append('a')
                out.append(_VOWEL[off])
                pending = False
            elif off in _NASAL:
                if pending:
                    out.append('a')
                    pending = False
                out.append('n')
            elif off in _FINAL:
                if pending:
                    out.append('a')
                out.append(_FINAL[off])
                pending = False
            elif 0x66 <= off <= 0x6F:
                pending = False
                out.append(chr(ord('0') + off - 0x66))
            # visarga, addak, avagraha, length marks: silent
            continue
        if ch in '‌‍':
            continue
        pending = False      # word-final schwa deletion
        last_cons = None
        out.append(ch)
    return ''.join(out)


_HAS_BRAHMIC = re.compile('[ऀ-ൿ]')

# --------------------------------------------------------------------------------------
# 2. Latin folding (accents, ligatures, typographic punctuation)
# --------------------------------------------------------------------------------------
_SPECIAL = str.maketrans({'ß': 'ss', 'æ': 'ae', 'Æ': 'ae', 'œ': 'oe', 'Œ': 'oe', 'ø': 'o',
                          'Ø': 'o', 'ł': 'l', 'Ł': 'l', 'đ': 'd', 'ı': 'i', '’': "'",
                          '‘': "'", '`': "'", '´': "'", '–': '-', '—': '-'})


def fold(s):
    if not s:
        return ''
    is_indic = bool(_HAS_BRAHMIC.search(s))
    if is_indic:
        s = transliterate_brahmic(s)
    s = s.translate(_SPECIAL)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.lower()


# --------------------------------------------------------------------------------------
# 3. Phonetic skeleton: script/typo-robust token key.
# "private" / "praivet" / "prywate" -> "prvt"
# --------------------------------------------------------------------------------------
_SKEL_SUBS = [('ph', 'f'), ('ck', 'k'), ('q', 'k'), ('x', 'ks'), ('w', 'v'), ('z', 'j'),
              ('c', 'k'), ('y', 'i')]
_DEDUP = re.compile(r'(.)\1+')
_VOWELS = re.compile(r'[aeiou]')
_H_AFTER = re.compile(r'(?<=[kgtdbjs])h')   # aspiration: kh->k, dh->d, sh->s ...


def skeleton(tok):
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = _H_AFTER.sub('', t)
    t = _DEDUP.sub(r'\1', t)
    head, tail = t[0], _VOWELS.sub('', t[1:])
    return head + tail


# --------------------------------------------------------------------------------------
# 4. Business name canonicalisation
# --------------------------------------------------------------------------------------
_URL = re.compile(r'(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|co|net|org|fr|io|biz|info|us|eu)(?:\.[a-z]{2})?\b')
_PHONE = re.compile(r'\+?\d[\d\s\-]{6,}\d')
_DOTTED = re.compile(r'\b(?:[a-z]\.){2,}[a-z]?\.?')
_MS = re.compile(r'\bm/s\.?\s*')
_NON_ALNUM = re.compile(r'[^a-z0-9]+')
_LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a', '$': 's'})
_MIXED = re.compile(r'[a-z@$]')

NAME_ABBR = {
    'pvt': 'private', 'prv': 'private', 'ltd': 'limited', 'ltda': 'limited', 'lmt': 'limited',
    'corp': 'corporation', 'co': 'company', 'cos': 'companies', 'inc': 'incorporated',
    'intl': 'international', 'bros': 'brothers', 'mfg': 'manufacturing', 'svc': 'services',
    'svcs': 'services', 'assoc': 'associates', 'mgmt': 'management', 'ent': 'enterprises',
    'ets': 'etablissements', 'n': 'and', 'et': 'and', 'y': 'and', 'und': 'and',
}
LEGAL_FORMS = {'private', 'public', 'limited', 'llp', 'llc', 'incorporated', 'corporation',
               'company', 'pllc', 'pc', 'lp', 'plc', 'lllp', 'pa', 'sarl', 'sas', 'sasu', 'sa',
               'eurl', 'sci', 'ei', 'snc', 'scop', 'selarl', 'gmbh', 'ag', 'bv', 'nv', 'opc'}
NAME_STOP = {'the', 'and', 'of', 'de', 'des', 'du', 'la', 'le', 'les', 'l', 'd', 'a', 'an', 'at',
             'for', 'et'}
LEGAL = LEGAL_FORMS | NAME_STOP
# skeletons of legal words, catches transliterated forms like "praivet limited"
_LEGAL_SKEL = {skeleton(w) for w in ('private', 'limited', 'corporation', 'incorporated', 'company')}


def _clean_name_token(t):
    if _MIXED.search(t) and any(ch.isdigit() for ch in t):
        t2 = t.translate(_LEET)
        if t2.isalpha():
            t = t2
    return NAME_ABBR.get(t, t)


def normalize_name(raw, country=None):
    """Returns (tokens, core_tokens, is_indic, has_domain).

    The record's own country label is dropped from the core name: sources inject it as noise
    ("ISLAMIQUE MUSIQUE (FRANCE)", "Newage (India) ..."). This uses the country *field*, so it
    works for any country string without enumerating countries.
    """
    if raw is None:
        raw = ''
    is_indic = bool(_HAS_BRAHMIC.search(raw))
    s = fold(raw)
    has_domain = False
    if '.' in s or 'www' in s:
        s2 = _URL.sub(lambda m: ' ' + m.group(1).replace('-', '') + ' ', s)
        has_domain = s2 != s
        s = s2
    if s.startswith('#'):
        has_domain = True
    s = _PHONE.sub(' ', s)
    s = _MS.sub(' ', s)
    s = _DOTTED.sub(lambda m: m.group(0).replace('.', ''), s)
    s = s.replace('&', ' and ').replace('+', ' plus ')
    s = s.replace("'", '')
    toks = []
    for t in _NON_ALNUM.split(s):
        if not t:
            continue
        t = _clean_name_token(t)
        if toks and toks[-1] == t:            # "VIDYALAYA VIDYALAYA"
            continue
        toks.append(t)
    cty = fold(country).strip() if country else ''
    core = [t for t in toks if t not in LEGAL and skeleton(t) not in _LEGAL_SKEL and t != cty]
    if not core:
        core = [t for t in toks if t != cty] or toks
    return toks, core, is_indic, has_domain


# --------------------------------------------------------------------------------------
# 5. Address canonicalisation
# --------------------------------------------------------------------------------------
ADDR_ABBR = {
    'st': 'st', 'str': 'st', 'street': 'st', 'saint': 'st', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue', 'avn': 'avenue',
    'dr': 'drive', 'drv': 'drive', 'ln': 'lane', 'ct': 'court', 'crt': 'court', 'blvd': 'boulevard',
    'bd': 'boulevard', 'bvd': 'boulevard', 'pl': 'place', 'cir': 'circle', 'hwy': 'highway',
    'pkwy': 'parkway', 'trl': 'trail', 'ter': 'terrace', 'terr': 'terrace', 'sq': 'square',
    'mt': 'mount', 'ft': 'fort', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'r': 'rue', 'rte': 'route', 'chem': 'chemin', 'imp': 'impasse', 'all': 'allee',
    'nr': 'near', 'opp': 'opposite', 'ngr': 'nagar', 'mkt': 'market', 'extn': 'extension',
    'ext': 'extension', 'sec': 'sector', 'apts': 'apartments', 'apt': 'apartment',
}
# unit designators / filler: vary freely between sources and carry no identity
ADDR_STOP = {'no', 'nos', 'number', 'h', 'hno', 'house', 'door', 'plot', 'flat', 'unit', 'suite',
             'ste', 'fl', 'floor', 'apartment', 'room', 'rm', 'po', 'box', 'bldg', 'building',
             'block', 'blk', 'office', 'shop', 'city', 'town', 'village', 'of', 'the', 'and',
             'null', 'none', 'na', 'nil', 'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd', 'bis',
             'th', 'nd', 'near', 'opposite', 'behind', 'c', 'o'}
_NUM = re.compile(r'\d+')
_ADDR_SPLIT = re.compile(r'[^a-z0-9]+')
_ORDINAL = re.compile(r'^(\d+)(st|nd|rd|th|er|e|eme)$')


_CODE = re.compile(r'(?:\b[a-z][/\-])?[a-z0-9]*\d[a-z0-9]*(?:[/\-][a-z0-9]+)*')
_CODE_SUFFIX = re.compile(r'^(\d+)(?:st|nd|rd|th|er|eme|bis|ter)$')


def address_codes(s):
    """Composite unit/house identifiers kept whole: 'C-303' -> 'c303', '2/1149/A61' -> '21149a61',
    '0508' -> '508'. Splitting these into loose digits loses exactly the part that tells two
    neighbouring units (siblings) apart."""
    out = []
    for m in _CODE.findall(s):
        c = m.replace('/', '').replace('-', '').lstrip('0')
        c = _CODE_SUFFIX.sub(r'\1', c)          # 265th -> 265, 45bis -> 45
        if len(c) >= 2:
            out.append(c)
    return list(dict.fromkeys(out))


def normalize_address(raw):
    """Returns (word_tokens, number_tokens, code_tokens)."""
    if not raw:
        return [], [], []
    s = fold(raw)
    s = s.replace('<null>', ' ').replace('n°', ' ').replace('°', ' ')
    codes = address_codes(s)
    words, nums = [], []
    for t in _ADDR_SPLIT.split(s):
        if not t:
            continue
        if t.isdigit():
            nums.append(t.lstrip('0') or '0')
            continue
        m = _ORDINAL.match(t)
        if m:                                   # "2nd" -> 2 ; "41st" -> 41
            nums.append(m.group(1).lstrip('0') or '0')
            continue
        if any(ch.isdigit() for ch in t):       # "b239", "55a", "74a"
            for n in _NUM.findall(t):
                nums.append(n.lstrip('0') or '0')
            alpha = ADDR_ABBR.get(re.sub(r'\d+', '', t), re.sub(r'\d+', '', t))
            if len(alpha) > 1 and alpha not in ADDR_STOP:
                words.append(alpha)
            continue
        t = ADDR_ABBR.get(t, t)
        if t in ADDR_STOP or len(t) == 1:
            continue
        words.append(t)
    # de-duplicate, keep order
    words = list(dict.fromkeys(words))
    nums = list(dict.fromkeys(nums))
    return words, nums, codes


def normalize_record(name, address, country=None):
    toks, core, is_indic, has_domain = normalize_name(name, country)
    aw, an, ac = normalize_address(address)
    return (' '.join(toks), ' '.join(core), ' '.join(skeleton(t) for t in core),
            ' '.join(aw), ' '.join(an), ' '.join(ac), is_indic, has_domain)
