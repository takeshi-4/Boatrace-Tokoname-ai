"""
How to use this script
(1) Download race result files by RaceResults.download()
(2) Manually extract text files from lzh files (e.g. 7-Zip)
(3) Move the extracted text files to ./data/results_race (NOT inside lzh/)
(4) loader.make_race_result_df() will parse the text files
"""

import os
import time
import random
import pandas as pd
import urllib.request
import urllib.error

current_dir = os.path.dirname(os.path.abspath(__file__))


class RaceResults:
    def __init__(self):
        # Example:
        # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.baseuri = "http://www1.mbrace.or.jp/od2/K/%s/k%s.lzh"
        self.id2index = None

        # Save directory
        self.save_dir = os.path.join(current_dir, "../../data/results_race/lzh")
        os.makedirs(self.save_dir, exist_ok=True)

        # User-Agent (some servers are stricter when UA is missing)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; BoatraceDownloader/1.0; +https://example.com)"
        }

    def _download_one(self, uri: str, savename: str, max_retries: int = 5) -> bool:
        """
        Returns True if downloaded successfully, False if skipped (e.g. 404).
        Retries on temporary errors.
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Build request with headers
                req = urllib.request.Request(uri, headers=self.headers)

                # Download
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()

                # Write file
                with open(savename, "wb") as f:
                    f.write(data)

                # Validate file size (0 bytes is suspicious)
                if os.path.getsize(savename) == 0:
                    try:
                        os.remove(savename)
                    except OSError:
                        pass
                    raise urllib.error.URLError("Downloaded file is 0 bytes")

                return True

            except urllib.error.HTTPError as e:
                # 404: no file (no race day) -> skip safely
                if e.code == 404:
                    print("No file (404), skip:", uri)
                    return False

                # 403/429: rate limit / forbidden -> wait and retry
                if e.code in (403, 429):
                    wait = min(60, 5 * attempt) + random.uniform(0.0, 1.5)
                    print(f"HTTP {e.code} (rate/forbidden). Wait {wait:.1f}s then retry: {uri}")
                    time.sleep(wait)
                    continue

                # Other HTTP errors: retry a bit, then give up
                wait = min(30, 2 * attempt) + random.uniform(0.0, 1.0)
                print(f"HTTP {e.code}. Wait {wait:.1f}s then retry({attempt}/{max_retries}): {uri}")
                time.sleep(wait)
                continue

            except urllib.error.URLError as e:
                # Network issues, DNS, reset, timeout, etc.
                wait = min(30, 2 * attempt) + random.uniform(0.0, 1.0)
                print(f"Network error. Wait {wait:.1f}s then retry({attempt}/{max_retries}): {uri} / {e}")
                time.sleep(wait)
                continue

            except Exception as e:
                # Unexpected error: retry a bit
                wait = min(30, 2 * attempt) + random.uniform(0.0, 1.0)
                print(f"Unexpected error. Wait {wait:.1f}s then retry({attempt}/{max_retries}): {uri} / {e}")
                time.sleep(wait)
                continue

        print("FAILED after retries:", uri)
        return False

    def download(self, start, end, sleep_sec: float = 1.0):
        period = pd.date_range(start, end)

        for date in period:
            dirname = date.strftime("%Y%m")
            lzhname = date.strftime("%y%m%d")
            uri = self.baseuri % (dirname, lzhname)
            savename = os.path.join(self.save_dir, f"{lzhname}.lzh")

            # Skip if already exists and is non-zero
            if os.path.exists(savename) and os.path.getsize(savename) > 0:
                continue

            print("Send request to", uri)
            self._download_one(uri, savename, max_retries=5)

            # polite wait (1 sec + small jitter)
            time.sleep(max(0.2, sleep_sec) + random.uniform(0.0, 0.5))


if __name__ == "__main__":
    r = RaceResults()

    # ✅ まずは小さく（成功確認用）
    # r.download("2020-01-01", "2020-03-31", sleep_sec=1.0)

    # ✅ 本番：必要な期間に変えて実行
    r.download("2019-12-10", "2019-12-31", sleep_sec=1.0)
