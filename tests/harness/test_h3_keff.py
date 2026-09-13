"""H3 per-material k_eff head — CSV, class defaults, mofdscribe provenance (DESIGN §8.1)."""

import csv
import tempfile
from pathlib import Path

import pytest

from harness.materials import MaterialParams
from harness.materials_k_eff import (
    K_EFF_BY_CLASS,
    K_EFF_GLOBAL_DEFAULT,
    assign_k_eff,
    assign_k_eff_to_materials,
    k_eff_from_mofdscribe,
    load_k_eff_csv,
    with_transport_k_eff_defaults,
)


def _mat(name="TestMOF", source="isodb", material_id="m1", material_class="MOF", k_eff=None):
    return MaterialParams(
        name=name,
        source=source,
        material_id=material_id,
        material_class=material_class,
        q_sat_kg_kg=0.40,
        q_st_j_kg=2.7e6,
        e_char_j_mol=6000.0,
        n_da=1.8,
        k_eff_w_m_k=k_eff,
    )


def test_load_k_eff_csv(tmp_path: Path):
    csv_path = tmp_path / "k_eff.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["material_id", "name", "source", "k_eff_w_m_k"])
        w.writeheader()
        w.writerow({"material_id": "m1", "name": "TestMOF", "source": "isodb", "k_eff_w_m_k": "0.42"})
        w.writerow({"material_id": "m2", "name": "Silica gel RD", "source": "anchor", "k_eff_w_m_k": "0.31"})
        w.writerow({"material_id": "m3", "name": "Bad", "source": "x", "k_eff_w_m_k": ""})
    table = load_k_eff_csv(csv_path)
    assert table["m1"] == pytest.approx(0.42)
    assert table["m2"] == pytest.approx(0.31)
    # Case-insensitive alias
    assert table["m1".lower()] == pytest.approx(0.42)
    # source:name alias
    assert table["isodb:TestMOF"] == pytest.approx(0.42)


def test_assign_k_eff_prefers_existing_fit():
    mat = _mat(k_eff=0.55)
    out = assign_k_eff(mat, k_eff_table={"m1": 0.42})
    assert out.k_eff_w_m_k == pytest.approx(0.55)
    assert out.transport_provenance == "none"  # unchanged, already had fit


def test_assign_k_eff_from_csv_provenance_mofdscribe():
    mat = _mat(material_id="m1", name="TestMOF")
    table = {"m1": 0.42, "TestMOF": 0.43, "isodb:TestMOF": 0.44}
    out = assign_k_eff(mat, k_eff_table=table)
    assert out.k_eff_w_m_k == pytest.approx(0.42)  # material_id wins
    assert out.transport_provenance == "mofdscribe"


def test_with_transport_k_eff_defaults_class_stratified():
    # No CSV — should pick class default with provenance default
    silica = _mat(name="SilicaA", material_id="s1", material_class="silica gel", k_eff=None)
    mof = _mat(name="MOF-5", material_id="m5", material_class="MOF", k_eff=None)
    zeo = _mat(name="Zeo13X", material_id="z1", material_class="zeolite", k_eff=None)
    carbon = _mat(name="AC", material_id="c1", material_class="carbon", k_eff=None)
    unknown = _mat(name="Foo", material_id="u1", material_class="unknown-xyz", k_eff=None)

    out_silica = with_transport_k_eff_defaults(silica)
    out_mof = with_transport_k_eff_defaults(mof)
    out_zeo = with_transport_k_eff_defaults(zeo)
    out_carbon = with_transport_k_eff_defaults(carbon)
    out_unknown = with_transport_k_eff_defaults(unknown)

    assert out_silica.k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["silica gel"])
    assert out_mof.k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["MOF"])
    assert out_zeo.k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["zeolite"])
    assert out_carbon.k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["carbon"])
    # Unknown class -> global default
    assert out_unknown.k_eff_w_m_k == pytest.approx(K_EFF_GLOBAL_DEFAULT)
    for o in (out_silica, out_mof, out_zeo, out_carbon, out_unknown):
        assert o.transport_provenance == "default"


def test_k_eff_from_mofdscribe_descriptors():
    mat = _mat()
    descriptors = {"density": 800.0, "void_fraction": 0.45}
    k_eff, prov = k_eff_from_mofdscribe(mat, descriptors)
    assert prov == "mofdscribe"
    assert 0.05 <= k_eff <= 1.5
    # Using descriptors path should be picked by assign_k_eff with provenance mofdscribe
    out = assign_k_eff(mat, mofdscribe_descriptors=descriptors)
    assert out.transport_provenance == "mofdscribe"
    assert out.k_eff_w_m_k == pytest.approx(k_eff)


def test_batch_helper():
    mats = [_mat(name=f"M{i}", material_id=f"m{i}", material_class="MOF") for i in range(3)]
    table = {"m1": 0.33}
    enriched = assign_k_eff_to_materials(mats, k_eff_table=table)
    # m0 and m2 get class default, m1 gets table
    assert enriched[1].k_eff_w_m_k == pytest.approx(0.33)
    assert enriched[1].transport_provenance == "mofdscribe"
    assert enriched[0].k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["MOF"])
    assert enriched[0].transport_provenance == "default"


def test_csv_helper_callable_from_rank_and_gui(tmp_path):
    # Simulate GUI/rank calling load_k_eff_csv + assign
    csv_path = tmp_path / "keff.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["material_id", "k_eff_w_m_k"])
        w.writeheader()
        for i in range(5):
            w.writerow({"material_id": f"m{i}", "k_eff_w_m_k": f"{0.10 + i*0.05:.2f}"})
    table = load_k_eff_csv(csv_path)
    mats = [_mat(material_id=f"m{i}", material_class="MOF") for i in range(5)]
    enriched = assign_k_eff_to_materials(mats, k_eff_table=table)
    for i, m in enumerate(enriched):
        assert m.k_eff_w_m_k == pytest.approx(0.10 + i * 0.05)


def test_with_transport_k_eff_defaults_more_granular_than_single():
    # Demonstrate that granular class defaults beat the old single 0.3
    silica = _mat(material_class="silica gel")
    zeo = _mat(material_class="zeolite")
    out_s = with_transport_k_eff_defaults(silica, fallback=0.30)
    out_z = with_transport_k_eff_defaults(zeo, fallback=0.30)
    # They must differ, proving granularity
    assert out_s.k_eff_w_m_k != pytest.approx(out_z.k_eff_w_m_k)
    assert out_s.k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["silica gel"])
    assert out_z.k_eff_w_m_k == pytest.approx(K_EFF_BY_CLASS["zeolite"])
