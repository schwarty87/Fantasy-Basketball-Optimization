import unittest
import numpy as np
import pandas as pd

from constants import FIX_PLAYER_NAMES
from utils import FixPlayerNames, remove_accents, normalize_player_name


class TestFixPlayerNames(unittest.TestCase):
    def test_remove_accents_standard_and_special_chars(self):
        """Verify remove_accents handles standard accents, non-decomposing chars, and non-strings."""
        self.assertEqual(remove_accents("Luka Dončić"), "Luka Doncic")
        self.assertEqual(remove_accents("Jonas Valančiūnas"), "Jonas Valanciunas")
        self.assertEqual(remove_accents("Dennis Schröder"), "Dennis Schroder")
        self.assertEqual(remove_accents("Nikola Đurišić"), "Nikola Durisic")
        self.assertEqual(remove_accents("Bogdan Bogdanović"), "Bogdan Bogdanovic")
        self.assertIsNone(remove_accents(None))
        self.assertEqual(remove_accents(123), 123)

    def test_normalize_player_name(self):
        """Verify normalize_player_name strips accents, punctuation, suffixes, and collapses spaces."""
        self.assertEqual(normalize_player_name("C.J. McCollum"), "cj mccollum")
        self.assertEqual(normalize_player_name("CJ McCollum"), "cj mccollum")
        self.assertEqual(normalize_player_name("Gary Trent Jr."), "gary trent")
        self.assertEqual(normalize_player_name("Gary Trent Jr"), "gary trent")
        self.assertEqual(normalize_player_name("Marcus Morris Sr."), "marcus morris")
        self.assertEqual(normalize_player_name("Trey Murphy III"), "trey murphy")
        self.assertEqual(normalize_player_name("Wade Baldwin IV"), "wade baldwin")
        self.assertEqual(normalize_player_name("De'Andre Hunter"), "deandre hunter")
        self.assertEqual(normalize_player_name("Trey McKinney-Jones"), "trey mckinney jones")
        self.assertEqual(normalize_player_name("  LeBron   James  "), "lebron james")
        self.assertEqual(normalize_player_name(None), "")
        self.assertEqual(normalize_player_name(np.nan), "")

    def test_fix_names_exact_and_case_insensitive(self):
        """Verify exact matches are preserved and case variations resolve to canonical casing."""
        canonical = ["LeBron James", "Stephen Curry", "Luka Doncic"]
        fixer = FixPlayerNames(compare_names=canonical)

        result = fixer.fix_names(["LeBron James", "stephen curry", "LUKA DONCIC"])
        self.assertEqual(list(result), ["LeBron James", "Stephen Curry", "Luka Doncic"])

    def test_fix_names_initials_and_punctuation(self):
        """Verify initial expansions and punctuation alignment to canonical names."""
        canonical = ["CJ McCollum", "P.J. Washington", "De'Andre Hunter"]
        fixer = FixPlayerNames(compare_names=canonical)

        # Scraped with opposite period conventions or missing apostrophes
        inputs = ["C.J. McCollum", "PJ Washington", "DeAndre Hunter"]
        result = fixer.fix_names(inputs)
        self.assertEqual(list(result), ["CJ McCollum", "P.J. Washington", "De'Andre Hunter"])

    def test_fix_names_suffix_matching(self):
        """Verify suffixes (Jr., III, etc.) are matched automatically when compare_names has the canonical form."""
        canonical = ["Gary Trent Jr.", "Trey Murphy III", "Kevin Knox II", "Marcus Morris Sr."]
        fixer = FixPlayerNames(compare_names=canonical)

        inputs = ["Gary Trent", "Trey Murphy", "Kevin Knox", "Marcus Morris"]
        result = fixer.fix_names(inputs)
        self.assertEqual(list(result), ["Gary Trent Jr.", "Trey Murphy III", "Kevin Knox II", "Marcus Morris Sr."])

    def test_fix_names_handles_none_and_nan(self):
        """Verify fix_names does not crash on missing names / None / NaN values."""
        canonical = ["LeBron James", "Stephen Curry"]
        fixer = FixPlayerNames(compare_names=canonical)

        raw = pd.Series(["LeBron James", None, np.nan, "stephen curry"])
        result = fixer.fix_names(raw)
        self.assertEqual(result[0], "LeBron James")
        self.assertTrue(pd.isna(result[1]))
        self.assertTrue(pd.isna(result[2]))
        self.assertEqual(result[3], "Stephen Curry")

    def test_fix_names_justin_anderson_not_conflated(self):
        """Verify Justin Anderson is never overwritten with Juan Toscano-Anderson."""
        canonical = ["Justin Anderson", "Juan Toscano-Anderson"]
        fixer = FixPlayerNames(compare_names=canonical)

        result = fixer.fix_names(["Justin Anderson", "Juan Toscano-Anderson"])
        self.assertEqual(list(result), ["Justin Anderson", "Juan Toscano-Anderson"])

    def test_fix_player_names_dict_integrity(self):
        """Verify FIX_PLAYER_NAMES has no self-mappings, no chains, and no dead keys."""
        # 1. No self-mappings
        self_mappings = [k for k, v in FIX_PLAYER_NAMES.items() if k == v]
        self.assertEqual(self_mappings, [], f"Self-mappings found: {self_mappings}")

        # 2. No circular/chained mappings (value should not be another key)
        chains = {k: v for k, v in FIX_PLAYER_NAMES.items() if v in FIX_PLAYER_NAMES}
        self.assertEqual(chains, {}, f"Chained mappings found: {chains}")

        # 3. No accented keys that can never match after remove_accents
        accented_keys = [k for k in FIX_PLAYER_NAMES if remove_accents(k) != k]
        self.assertEqual(accented_keys, [], f"Accented keys that can never match: {accented_keys}")

        # 4. Justin Anderson must NOT be mapped to Juan Toscano-Anderson
        self.assertNotIn("Justin Anderson", FIX_PLAYER_NAMES)

    def test_fix_names_first_name_prefix_diminutives(self):
        """Verify first-name prefix variations (Dom -> Dominick) resolve when surname is unique in compare_names."""
        canonical = ["Dominick Barlow", "Alexandre Sarr", "Herbert Jones"]
        fixer = FixPlayerNames(compare_names=canonical)

        inputs = ["Dom Barlow", "Alex Sarr", "Herb Jones"]
        result = fixer.fix_names(inputs)
        self.assertEqual(list(result), ["Dominick Barlow", "Alexandre Sarr", "Herbert Jones"])


if __name__ == "__main__":
    unittest.main()
