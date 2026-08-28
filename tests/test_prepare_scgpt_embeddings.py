import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_scgpt_embeddings.py"
SPEC = importlib.util.spec_from_file_location("prepare_scgpt_embeddings", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hgnc_mapping_prefers_approved_symbol_and_supports_previous_symbol(tmp_path):
    hgnc = tmp_path / "hgnc.tsv"
    hgnc.write_text(
        "symbol\tprev_symbol\talias_symbol\tensembl_gene_id\n"
        "AARS1\tAARS\tALIAS_A\tENSG1\n"
        "CURRENT\tOLD\tALIAS_B\tENSG2\n"
    )
    observed = MODULE.load_hgnc_symbols(
        hgnc, {"AARS1": 0, "AARS": 1, "OLD": 2}
    )
    assert observed == {
        "ENSG1": ("AARS1", "AARS1"),
        "ENSG2": ("CURRENT", "OLD"),
    }
