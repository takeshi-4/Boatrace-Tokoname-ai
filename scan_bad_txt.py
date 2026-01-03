from pathlib import Path
import re

base = Path(r"C:\Users\Angel\Desktop\Boatrace-Tokoname-ai\boatrace-master\data\results_race")
txts = sorted(base.glob("K*.TXT"))

date_re = re.compile(r"\d{4}/\s*\d{1,2}/\s*\d{1,2}")
startk_re = re.compile(r"^STARTK\b")

bad = []
for p in txts:
    text = p.read_text(encoding="cp932", errors="replace")
    lines = text.splitlines()

    if not lines:
        bad.append((p.name, "empty"))
        continue

    if not startk_re.search(lines[0]):
        bad.append((p.name, f"missing STARTK in line0: '{lines[0]}'"))
        continue

    # ヘッダ先頭40行くらいに日付があるか（あなたの例だと8行目にある）
    head = lines[:60]
    found_date = None
    for ln in head:
        m = date_re.search(ln)
        if m:
            found_date = m.group(0).replace(" ", "")
            break

    if not found_date:
        bad.append((p.name, "no date pattern in first 60 lines"))
        continue

print(f"TXT count: {len(txts)}")
print(f"BAD count: {len(bad)}")
for name, reason in bad[:100]:
    print("\n====", name, "====")
    print(reason)

if bad:
    print("\nFirst bad file:", bad[0][0])
else:
    print("\nNo bad files detected by this check.")
