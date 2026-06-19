#!/usr/bin/env python3
import glob, os, sys
import recover_infinity_json as R

base = sys.argv[1] if len(sys.argv) > 1 else "output/production"
fails = []
for d in sorted(glob.glob(base + "/*.pdf")):
    rj = d + "/result.json"
    if not os.path.exists(rj):
        continue
    raw = open(rj, encoding="utf-8").read()
    try:
        R.recover(raw)
    except Exception as e:
        fails.append((os.path.basename(d), str(e)[:50], len(raw)))
print("recover() failures:", len(fails))
for n, e, sz in fails:
    print("  ", n, "|", e, "| bytes", sz)

# text fidelity on a doc that originally had bad escapes/quotes
for name in ["ColonialOfficeList1890.pdf", "ColonialOfficeList1894.pdf"]:
    p = base + "/" + name + "/result.json"
    if not os.path.exists(p):
        continue
    pages = R.recover(open(p, encoding="utf-8").read())
    blk = [b for pg in pages for b in pg]
    chars = sum(len(b["text"]) for b in blk)
    print("\n%s recovered: pages=%d blocks=%d text_chars=%d" % (name, len(pages), len(blk), chars))
    s = [b["text"] for b in blk if "circ" in b["text"] or chr(34) in b["text"]]
    if s:
        print("  sample:", repr(s[0][:140]))
