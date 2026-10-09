"""Download the public restaurant food-wastage CSV used for the ML demo.

Dataset source repository: https://github.com/Ashutosh-dutta07/Smart-Food-Waste-Intelligence
Raw CSV: https://raw.githubusercontent.com/Ashutosh-dutta07/Smart-Food-Waste-Intelligence/main/food_wastage_data.csv
Please review the source repository's current terms/licensing before redistribution.
"""
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/Ashutosh-dutta07/Smart-Food-Waste-Intelligence/main/food_wastage_data.csv"
OUT = Path(__file__).resolve().parent / "data" / "food_wastage_data.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "SmartWasteAI-educational-project/1.0"})
    print("Downloading public restaurant food-wastage dataset...")
    try:
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if len(content) < 1000 or b"Wastage Food Amount" not in content[:1000]:
            raise ValueError("The downloaded file did not look like the expected CSV.")
        OUT.write_bytes(content)
        print(f"Saved: {OUT}")
        print(f"Size: {len(content):,} bytes")
        print("Next step: python train_model.py")
    except Exception as exc:
        print("Dataset download failed:", exc)
        print("You can download the CSV manually from:")
        print(URL)
        print(f"Save it as: {OUT}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
