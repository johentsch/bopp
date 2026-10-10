"""Parse DCML harmony labels and generate their JSON Schema pattern.

The reference expression is ms3's ``DCML_DOUBLE_REGEX``
(https://github.com/johentsch/ms3, GNU General Public License v3, see
ms3's LICENSE.txt), kept verbatim below as ``MS3_DCML_DOUBLE_REGEX``. It
supports an optional alternative after a hyphen and accepts the empty
string, since every part of it is optional.

ms3's pattern suffers from catastrophic backtracking on labels it
otherwise accepts: the accidental group ``(b*|\\#*)`` can match the empty
string through either branch, and its numeral chains (localkey, pedal,
relativeroot, and their ``...2`` twins) let an optional trailing slash
make a run of numerals ambiguous to split, e.g. ``"III"`` as ``III``,
``II``+``I``, ``I``+``II``, or ``I``+``I``+``I``. Both multiply the number
of ways to match (or fail to match) a numeral run, which is exponential in
run length. ``DCML_REGEX`` below rewrites both constructs so matching
stays linear in the input length; the one observable behavioural change is
that it rejects slash-free numeral run-ons ms3 happens to accept, such as
``"IIIIII[II"``.

``DCML_REGEX`` keeps Python ``re`` semantics and is meant for parsing via
its named groups. ``DCML_PATTERN``, used for the JSON Schema, additionally
pins down three constructs that differ between Python ``re`` and the
ECMA-262 regex dialect used by JavaScript validators: ``\\d`` (Unicode
digits in Python, ASCII-only in ECMA-262), ``.`` (Python excludes only
``\\n``, ECMA-262 excludes all line terminators), and the final ``$``
(Python also matches just before one trailing newline).
"""

import re

# Copied verbatim from ms3's DCML_DOUBLE_REGEX in src/ms3/utils/constants.py.
# Source: https://github.com/johentsch/ms3 (GNU General Public License v3,
# see ms3's LICENSE.txt).
MS3_DCML_DOUBLE_REGEX = re.compile(
    r"""
                                ^(?P<first>
                                  (\.?
                                    ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\.)?
                                    ((?P<pedal>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\[)?
                                    (?P<chord>
                                        (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form>(%|o|\+|M|\+M))?
                                        (?P<figbass>(7|65|43|42|2|64|6))?
                                        (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend>\])?
                                  )?
                                  (\|(?P<cadence>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?
                                  (?P<phraseend>(\\\\|\}\{|\{|\})
                                  )?
                                 )
                                 (-
                                  (?P<second>
                                    ((?P<globalkey2>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\.)?
                                    ((?P<pedal2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\[)?
                                    (?P<chord2>
                                        (?P<numeral2>(b*|\#*)(
                                        VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form2>(%|o|\+|M|\+M))?
                                        (?P<figbass2>(7|65|43|42|2|64|6))?
                                        (\((?P<changes2>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend2>\])?
                                  )?
                                  (\|(?P<cadence2>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?
                                  (?P<phraseend2>(\\\\|\}\{|\{|\})
                                  )?
                                 )?
                                $
                                """,
    re.VERBOSE,
)

_NUMERAL = r"(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)"
_PLUS_CHAIN = r"((b+|\#+)?" + _NUMERAL + r"/?)+"
_STAR_CHAIN = r"((b+|\#+)?" + _NUMERAL + r"/?)*"
_CHAIN = r"(b+|\#+)?" + _NUMERAL + r"(/(b+|\#+)?" + _NUMERAL + r")*/?"


def _build_dcml_regex() -> re.Pattern[str]:
    """De-ambiguate ms3's numeral chains so matching them stays linear."""
    pattern = MS3_DCML_DOUBLE_REGEX.pattern.replace(r"(b*|\#*)", r"(b+|\#+)?")
    plus_count = pattern.count(_PLUS_CHAIN)
    star_count = pattern.count(_STAR_CHAIN)
    assert (plus_count, star_count) == (4, 2), (
        "ms3's DCML_DOUBLE_REGEX source has changed: expected 4 '+' numeral "
        f"chains (localkey, pedal, localkey2, pedal2) and 2 '*' numeral "
        f"chains (relativeroot, relativeroot2), found {plus_count} and "
        f"{star_count}"
    )
    pattern = pattern.replace(_PLUS_CHAIN, _CHAIN)
    pattern = pattern.replace(_STAR_CHAIN, "(" + _CHAIN + ")?")
    return re.compile(pattern, re.VERBOSE)


DCML_REGEX = _build_dcml_regex()


def _check_flattenable(pattern: str) -> None:
    """Reject re.VERBOSE source that blind whitespace-stripping would corrupt.

    Inside a character class, re.VERBOSE whitespace is literal, not
    ignored, so stripping it would change the class's meaning. Likewise an
    unescaped '#' starts a comment running to end of line, which naive
    stripping does not emulate. Neither occurs in the DCML pattern (ms3
    escapes '#' as '\\#' and uses no character-class whitespace).
    """
    in_class = False
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "\\":
            i += 2
            continue
        if in_class:
            if char == "]":
                in_class = False
            elif char.isspace():
                raise ValueError(
                    "portable_pattern cannot flatten a character class "
                    "containing literal whitespace"
                )
        elif char == "[":
            in_class = True
        elif char == "#":
            raise ValueError(
                "portable_pattern cannot flatten a re.VERBOSE comment "
                "('#' outside a character class)"
            )
        i += 1


def portable_pattern(regex: re.Pattern[str]) -> str:
    """Convert a DCML verbose expression to an ECMA-262 portable pattern.

    Removes re.VERBOSE-only formatting whitespace, named groups (including
    numbered alternative groups), and the Python-only escape before hash
    signs; then narrows three constructs that mean different things in
    Python ``re`` and ECMA-262: ``\\d`` (ASCII-only in ECMA-262), ``.``
    (ECMA-262 also excludes \\r and the two Unicode line separators), and
    the trailing ``$`` (Python also matches before one trailing newline).
    """
    _check_flattenable(regex.pattern)
    pattern = re.sub(r"\s+", "", regex.pattern)
    pattern = re.sub(r"\(\?P<[A-Za-z_]\w*>", "(", pattern)
    pattern = pattern.replace(r"\#", "#")
    pattern = pattern.replace(r"\d", "[0-9]")
    # A non-raw literal (unlike a raw one) keeps  /  as six literal
    # characters rather than decoding them to the actual code points, and a
    # function replacement (unlike a template string) is inserted verbatim
    # by re.sub instead of having its own backslash escapes reinterpreted.
    pattern = re.sub(r"(?<!\\)\.", lambda _: "[^\\n\\r\\u2028\\u2029]", pattern)
    assert pattern.endswith("$")
    pattern = pattern[: -len("$")] + r"(?![\s\S])"
    return pattern


DCML_PATTERN = portable_pattern(DCML_REGEX)
