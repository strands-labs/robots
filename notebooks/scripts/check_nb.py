"""Print every output of a notebook that is an error, or a stderr stream (warnings, tracebacks)."""
import json, sys
bad = 0
for f in sys.argv[1:]:
    with open(f) as fh:
        nb = json.load(fh)
    for i, c in enumerate(nb["cells"]):
        for o in c.get("outputs", []):
            if o["output_type"] == "error" or (o["output_type"] == "stream" and o["name"] == "stderr"):
                bad += 1
                print(f"{f} cell {i}: {o.get('ename') or 'stderr'}: {(''.join(o.get('text', '')) or o.get('evalue', ''))[:400]}")
print("bad outputs:", bad)
sys.exit(1 if bad else 0)
