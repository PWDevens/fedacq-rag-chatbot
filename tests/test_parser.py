## tests/test_parser.py

from rag.retrieval.parser_dita import infer_part_subpart_section

def test_infer_section():
    part, subpart, section, clause = infer_part_subpart_section("1.101.dita")
    assert part == "1"
    assert section == "1.101"

