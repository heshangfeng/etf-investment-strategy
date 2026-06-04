import json

# portfolio.json
pf = json.load(open("data/portfolio.json", "r", encoding="utf-8"))
pf_cash = pf["cash"]
pf_hv = sum(h["shares"] * h["avg_cost"] for h in pf["holdings"])
pf_total = pf_cash + pf_hv
print(f"[portfolio.json]  cash={pf_cash:.2f}  holdings={pf_hv:.2f}  total={pf_total:.2f}")

# trade_log.json
tl = json.load(open("data/trade_log.json", "r", encoding="utf-8"))
last = tl[-1]
print(f"[trade_log.json]  cash={last['cash']:.2f}  mkt_val={last['market_value']:.2f}  total={last['total']:.2f}  trades={len(last['trades']['buys'])}buys/{len(last['trades']['sells'])}sells")

# performance.json
perf = json.load(open("data/performance.json", "r", encoding="utf-8"))
print(f"[performance.json]  init_cap={perf['initial_capital']:.2f}  cur_val={perf['current_value']:.2f}  return={perf['total_return_pct']}%")

# Check consistency
print()
if abs(perf["current_value"] - pf_total) > 0.01:
    print(f"❌ performance.current_value ({perf['current_value']:.2f}) != portfolio total ({pf_total:.2f})")
else:
    print(f"✅ performance.current_value == portfolio total ({pf_total:.2f})")

if abs(last["total"] - pf_total) > 0.01:
    print(f"❌ trade_log last total ({last['total']:.2f}) != portfolio total ({pf_total:.2f})")
else:
    print(f"✅ trade_log last total == portfolio total ({pf_total:.2f})")

# Check if trade_log snapshots make sense
for i, snap in enumerate(tl):
    expected = snap["cash"] + snap["market_value"]
    if abs(expected - snap["total"]) > 0.01:
        print(f"❌ snapshot[{i}] cash+mkt ({expected:.2f}) != total ({snap['total']:.2f})")
    else:
        print(f"✅ snapshot[{i}] {snap['date']}: cash+mkt=total ({snap['total']:.2f})")
