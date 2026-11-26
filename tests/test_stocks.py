import numpy as np
import pandas as pd

from dhs import sim, stocks


def test_implied_vol_round_trip():
    for kind in ("call", "put"):
        p = float(sim.black76(1000.0, 980.0, 0.08, 0.065, 0.32, kind))
        assert np.isclose(stocks.implied_vol(p, 1000.0, 980.0, 0.08, 0.065, kind), 0.32, atol=1e-6)


def test_symbol_chains_and_atm_case_from_a_small_option_file(tmp_path, monkeypatch):
    (tmp_path / "_ref").mkdir()
    (tmp_path / "_ref" / "symbolchange.csv").write_text(
        "NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\n"
        "X,AAA,BBB,01-JAN-2020\nX,BBB,CCC,01-JAN-2022\n")
    monkeypatch.setenv("DHS_LAKE_DIR", str(tmp_path))
    assert stocks.current_symbols() == {"AAA": "CCC", "BBB": "CCC"}
    d, e, F, T = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), 1000.0, 29 / 365
    rows = [{"date": d, "symbol": "AAA", "expiry": e, "strike": np.nan, "type": "FUT",
             "price": F, "settle": F, "contracts": 10, "oi": 1}]
    for K in (980.0, 1000.0, 1020.0):
        for typ, kind in (("CE", "call"), ("PE", "put")):
            rows.append({"date": d, "symbol": "AAA", "expiry": e, "strike": K, "type": typ,
                         "price": float(sim.black76(F, K, T, 0.065, 0.3, kind)),
                         "settle": 0.0, "contracts": 5, "oi": 1})
    (tmp_path / "data" / "options").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "options" / "stock_options.parquet")
    monkeypatch.setenv("DHS_IV_DIR", str(tmp_path))
    c = stocks.atm_cases()
    assert len(c) == 1 and c["K"].iloc[0] == 1000.0 and np.isclose(c["vol"].iloc[0], 0.3, atol=1e-6)
