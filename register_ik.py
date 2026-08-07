#!/usr/bin/env python3
"""Register the IK data pull alongside the existing cohorts.

Two places, matching how every other site is recorded:

  ~/.config/boost_backtest/sites_all.json   site registry — tag, base, token, timezone offset,
                                            system, pipeline, active flag, note
  oref.user_profiles                        per-user metadata derived from the Nightscout profile
                                            and the extracted decision stream

Nothing here is printed or written outside those two locations, and neither is in a git repository.
"""
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import psycopg2

REG = os.path.expanduser("~/.config/boost_backtest/sites_all.json")
DSN = "dbname=oref host=127.0.0.1 port=5432"
SITES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ik_sites.json")))
NOTE = "Insulin-kinetics controlled test, added 2026-08-05; known setups held by the investigator"


def ns(base, token, path, **p):
    if token:
        p["token"] = token
    req = urllib.request.Request(f"{base.rstrip('/')}{path}?{urllib.parse.urlencode(p)}",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def profile_for(uid, base, token):
    pr = ns(base, token, "/api/v1/profile.json", count=1)[0]
    store = pr.get("store") or {}
    blk = None
    for _, v in store.items():
        if isinstance(v, dict):
            blk = v
            break
    ds = ns(base, token, "/api/v1/devicestatus.json", count=1)[0]

    def sched_mean(key):
        s = (blk or {}).get(key) or []
        vals = [float(x["value"]) for x in s if isinstance(x, dict) and "value" in x]
        return vals

    basal = sched_mean("basal")
    isf = sched_mean("sens")
    cr = sched_mean("carbratio")
    tgt = sched_mean("target_low")
    units = (blk or {}).get("units") or pr.get("units") or "mg/dl"
    is_mmol = str(units).lower().startswith("mmol")

    conn = psycopg2.connect(DSN)
    d = pd.read_sql("""SELECT count(*) n, avg(sug_tdd) tdd FROM boost_decisions
                       WHERE user_id=%s""", conn, params=(uid,)) if False else pd.read_sql(
        """SELECT count(*) n FROM boost_decisions WHERE user_id=%s""", conn, params=(uid,))
    tre = pd.read_sql("""SELECT ts_utc, insulin FROM boost_treatments
                         WHERE user_id=%s AND insulin>0""", conn, params=(uid,))
    conn.close()
    days = max((tre.ts_utc.max() - tre.ts_utc.min()).days, 1) if len(tre) else 1
    mean_tdd = round(float(tre.insulin.sum()) / days, 1) if len(tre) else None

    prof = {
        "user_id": uid,
        "device": ds.get("device", "?"),
        "units": units,
        "is_mmol": is_mmol,
        "dia": (blk or {}).get("dia"),
        "timezone": (blk or {}).get("timezone"),
        "carbs_hr": (blk or {}).get("carbs_hr"),
        "n_decisions": int(d.n.iloc[0]),
        "mean_tdd": mean_tdd,
        "profile_n_basal_segments": len(basal),
        "profile_mean_basal": round(float(np.mean(basal)), 3) if basal else None,
        "profile_max_basal": round(float(np.max(basal)), 3) if basal else None,
        "profile_total_basal": round(float(np.sum(basal)), 1) if basal else None,
        "profile_isf_mean_mgdl": round(float(np.mean(isf)) * (18.0 if is_mmol else 1.0), 1) if isf else None,
        "profile_cr_mean": round(float(np.mean(cr)), 1) if cr else None,
        "profile_target_low_mgdl": round(float(np.mean(tgt)) * (18.0 if is_mmol else 1.0), 1) if tgt else None,
        "source": "insulin-kinetics controlled test",
    }
    return prof


if __name__ == "__main__":
    reg = json.load(open(REG))
    existing = {s["tag"] for s in reg["sites"]}
    added = 0
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    for s in SITES:
        uid = s["id"]
        prof = profile_for(uid, s["url"], s["token"])
        tzoff = None
        try:
            import zoneinfo, datetime as dt
            tzoff = int(dt.datetime.now(zoneinfo.ZoneInfo(prof["timezone"])).utcoffset().total_seconds() // 3600)
        # except-ok: a missing or unrecognised profile timezone leaves tzoff None, which the
        # registry records as unknown. Nothing downstream computes on it, so failing here would
        # abort a registration over a cosmetic field.
        except Exception:                                        # noqa: BLE001
            pass
        if uid not in existing:
            reg["sites"].append({
                "tag": uid, "base": s["url"], "token": s["token"],
                "tz_offset_hours": tzoff, "boost": None,
                "pipeline": "insulin-kinetics", "active": True,
                "system": prof["device"], "note": NOTE,
            })
            added += 1
        cur.execute("""INSERT INTO user_profiles (user_id, version, profile)
                       VALUES (%s,%s,%s)
                       ON CONFLICT (user_id) DO UPDATE SET version=EXCLUDED.version,
                                                           profile=EXCLUDED.profile""",
                    (uid, "ik1", json.dumps(prof)))
        print(f"  {uid}: profile stored ({prof['device']}, {prof['timezone']}, "
              f"dia={prof['dia']}, tdd~{prof['mean_tdd']})")
    conn.commit(); conn.close()
    json.dump(reg, open(REG, "w"), indent=1)
    print(f"\nregistry: {added} sites appended, {len(reg['sites'])} total")
