import os, sys, zipfile, requests

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_zip = os.path.join(data_dir, "uci_har.zip")

    print(f"Downloading UCI-HAR from:\n  {URL}")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with open(out_zip, "wb") as f:
        f.write(r.content)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(out_zip, "r") as zf:
        zf.extractall(data_dir)

    print("✅ Done. Check for folder: data/UCI HAR Dataset/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
