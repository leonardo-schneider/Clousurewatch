import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

metros = [
    ("Indianapolis",  "IN"),
    ("Tucson",        "AZ"),
    ("Nashville",     "TN"),
    ("New Orleans",   "LA"),
    ("Saint Louis",   "MO"),
    ("Reno",          "NV"),
    ("Boise",         "ID"),
    ("Edmonton",      "AB"),
]

def run_metro(city, state):
    start = datetime.now()
    cmd = [sys.executable, "09_multi_metro.py", "--city", city, "--state", state]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = (datetime.now() - start).seconds
    status = "OK" if result.returncode == 0 else "FAIL"
    return city, status, elapsed, result.stdout[-500:], result.stderr[-300:]

print(f"Starting {len(metros)} metros in parallel...")
print("=" * 55)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_metro, c, s): c for c, s in metros}
    for future in as_completed(futures):
        city, status, elapsed, stdout, stderr = future.result()
        print(f"\n[{status}] {city} ({elapsed}s)")
        print(f"  Last output: {stdout.strip()[-200:]}")
        if stderr:
            print(f"  STDERR: {stderr.strip()[-150:]}")

print("\n" + "=" * 55)
print("All metros done. Running backfill...")
result = subprocess.run(
    [sys.executable, "09_multi_metro.py", "--backfill-only"],
    capture_output=True, text=True, encoding="utf-8"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-300:])
print("Done. Ready for 10_lomo_cv.py")
