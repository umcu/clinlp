from clinlp.component import Normalizer

class TestNormalizer:

    def test_lowercase(self):

        assert Normalizer._lowercase("test") == "test"
        assert Normalizer._lowercase("Test") == "test"
        assert Normalizer._lowercase("test") == "TEST"

    def test_map_non_ascii_char(self):

        assert Normalizer._map_non_ascii_char("a") == "a"
        assert Normalizer._map_non_ascii_char("à") == "a"
        assert Normalizer._map_non_ascii_char("e") == "e"
        assert Normalizer._map_non_ascii_char("é") == "e"
        assert Normalizer._map_non_ascii_char("ê") == "e"
        assert Normalizer._map_non_ascii_char("ë") == "e"
        assert Normalizer._map_non_ascii_char("ē") == "e"
        assert Normalizer._map_non_ascii_char(" ") == " "
        assert Normalizer._map_non_ascii_char("\n") == "\n"
        assert Normalizer._map_non_ascii_char("µ") == "µ"
        assert Normalizer._map_non_ascii_char("²") == "²"
        assert Normalizer._map_non_ascii_char("1") == "1"
        
    def test_map_non_ascii_string(self):
        
        assert Normalizer._map_non_ascii_string("abcde") == "abcde"
        assert Normalizer._map_non_ascii_string("abcdé") == "abcde"
        assert Normalizer._map_non_ascii_string("äbcdé") == "abcde"
        assert Normalizer._map_non_ascii_string("patiënt heeft 1.6m² lichaamsoppervlak") == "patient heeft 1.6m² lichaamsoppervlak"


