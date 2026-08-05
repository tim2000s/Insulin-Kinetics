#!/usr/bin/env python3
"""Validate the wider-archive dose reconstruction against a known treatment stream.

The wider archive recovers doses from jumps in `bolusinsulin` (cumulative delivered within the DIA
window) and fits them against `bolusiob`. The fits come out 3-10x worse than the same estimator
achieves on the main cohort, and there is no way to tell why from inside that archive because it
holds no treatment record.

The main cohort can settle it. Its Nightscout devicestatus carries `bolusinsulin` too — it is simply
not extracted into the database — so pulling it directly gives, for the same users and the same
period, BOTH the reconstruction input and the ground-truth dose series.
"""
import datetime as dt
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import psycopg2

DSN = "dbname=oref host=127.0.0.1 port=5432"


def sites():
    out = {}
    reg = json.load(open("/Users/timstreet/.config/boost_backtest/sites.json"))["sites"]
    for s in reg:
        out["tim" if s["tag"] == "self" else s["tag"]] = (s["base"], s.get("token", ""))
    j = json.load(open("/Users/timstreet/SID-evaluation/multi_user/sites.json"))[0]
    out["J"] = (j["url"], j["token"])
    return out


def pull(base, token, days=30):
    since = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, before = [], None
    for _ in range(40):
        p = {"count": 1000, "token": token, "find[created_at][$gte]": since}
        if before:
            p["find[created_at][$lt]"] = before
        req = urllib.request.Request(
            f"{base.rstrip('/')}/api/v1/devicestatus.json?{urllib.parse.urlencode(p)}",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            batch = json.loads(urllib.request.urlopen(req, timeout=90).read())
        except Exception as e:                                    # noqa: BLE001
            print(f"    fetch stopped: {type(e).__name__}")
            break
        if not batch:
            break
        for b in batch:
            iob = (b.get("openaps") or {}).get("iob") or {}
            if isinstance(iob, list):
                iob = iob[0] if iob else {}
            if "bolusinsulin" in iob and "bolusiob" in iob:
                rows.append((b.get("created_at"), iob["bolusinsulin"], iob["bolusiob"]))
        before = batch[-1].get("created_at")
        if len(batch) < 1000:
            break
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["ts", "bi", "iob"])
    # resolution-independent: astype("int64") is us on this pandas, ns on others
    ts = pd.to_datetime(d.ts, utc=True, format="mixed")
    d["t"] = (ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return d.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def truth(user, t0, t1):
    conn = psycopg2.connect(DSN)
    d = pd.read_sql("""SELECT ts_utc, insulin FROM boost_treatments
                       WHERE user_id=%s AND insulin>0 ORDER BY ts_utc""", conn, params=(user,))
    conn.close()
    ts = pd.to_datetime(d.ts_utc, utc=True)
    d["t"] = (ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return d[(d.t >= t0) & (d.t <= t1)]


if __name__ == "__main__":
    S = sites()
    print(f"{'user':<6}{'samples':>9}{'recon U':>10}{'true U':>9}{'ratio':>8}"
          f"{'quantised':>11}{'matched doses':>15}")
    for u in ("J", "tim", "B", "D"):
        if u not in S:
            continue
        d = pull(*S[u], days=30)
        if d is None or len(d) < 500:
            print(f"{u:<6}  no bolusinsulin in devicestatus"); continue
        t = d.t.values
        bi = d.bi.values.astype(float)
        gap = np.diff(t)
        ok = (gap > 0) & (gap <= 900)
        dose_t, dose_u = t[1:][ok], np.maximum(np.diff(bi)[ok], 0.0)
        m = dose_u > 1e-9
        dose_t, dose_u = dose_t[m], dose_u[m]
        tr = truth(u, t.min(), t.max())
        q = np.mean(np.abs(dose_u / 0.05 - np.round(dose_u / 0.05)) < 1e-6) if len(dose_u) else 0
        # how many true doses have a reconstructed dose within 5 minutes?
        matched = 0
        if len(dose_t) and len(tr):
            for tt in tr.t.values:
                if np.min(np.abs(dose_t - tt)) <= 300:
                    matched += 1
        print(f"{u:<6}{len(d):>9,}{dose_u.sum():>10.0f}{tr.insulin.sum():>9.0f}"
              f"{dose_u.sum()/max(tr.insulin.sum(),1e-9):>8.2f}{100*q:>10.0f}%"
              f"{f'{matched}/{len(tr)}':>15}")
