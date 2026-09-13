"""Tests for the IZA-SC framework parser (offline, synthetic HTML)."""

from iza_frameworks import parse_framework_page

FAU_HTML = b"""<html><body>
Volume = 14428.8 &Aring; 3 &nbsp; Framework density (FD Si ): 13.3 T/1000 &Aring; 3
&nbsp; Essential rings: [12 1 &middot;6 2 &middot;4 2 ]
Channel dimensionality: Topological (pore opening &gt; 6-ring): 3-dimensional
Maximum diameter of a sphere: that can be included 11.24&nbsp;&Aring;
that can diffuse along a: &nbsp; 7.35&nbsp;&Aring; b: &nbsp; 7.35&nbsp;&Aring;
c: &nbsp; 7.35&nbsp;&Aring; &nbsp; Accessible volume: 27.42 &nbsp; %
</body></html>"""


def test_parse_fau_shape():
    row = parse_framework_page(FAU_HTML, "FAU", 93)
    assert row["code"] == "FAU" and row["page_id"] == 93
    assert row["fd_si"] == 13.3
    assert row["lcd_included_a"] == 11.24
    assert (row["pld_diffuse_a"], row["pld_diffuse_b"],
            row["pld_diffuse_c"]) == (7.35, 7.35, 7.35)
    assert row["pld_min_a"] == 7.35
    assert row["accessible_vol_pct"] == 27.42
    assert row["channel_dim"] == 3


def test_parse_absent_fields_are_none():
    row = parse_framework_page(b"<html><body>nothing here</body></html>", "XXX", 1)
    assert row["lcd_included_a"] is None and row["pld_min_a"] is None
    assert row["fd_si"] is None and row["channel_dim"] is None
