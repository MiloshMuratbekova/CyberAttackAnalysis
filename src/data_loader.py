"""CICIDS2017 dataset loader with automatic download from HuggingFace."""

import subprocess
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "input"

# HuggingFace mirror of CICIDS2017 MachineLearningCSV files
HF_BASE = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"

CICIDS_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# Map granular labels to higher-level attack categories
ATTACK_CATEGORY_MAP = {
    "BENIGN": "benign",
    # DoS
    "DoS slowloris": "dos",
    "DoS Slowhttptest": "dos",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "Heartbleed": "dos",
    # DDoS
    "DDoS": "ddos",
    # Brute Force
    "FTP-Patator": "brute_force",
    "SSH-Patator": "brute_force",
    # Web Attacks (various encodings of en-dash found in different CSV sources)
    "Web Attack \u2013 Brute Force": "web_attack",
    "Web Attack \u2013 XSS": "web_attack",
    "Web Attack \u2013 Sql Injection": "web_attack",
    "Web Attack \ufffd Brute Force": "web_attack",
    "Web Attack \ufffd XSS": "web_attack",
    "Web Attack \ufffd Sql Injection": "web_attack",
    # Infiltration
    "Infiltration": "infiltration",
    # Botnet
    "Bot": "botnet",
    # Port Scan
    "PortScan": "portscan",
}



def download_cicids(force: bool = False) -> Path:
    """Download CICIDS2017 CSV files from HuggingFace if not present."""
    marker = DATA_DIR / ".cicids_complete"

    if marker.exists() and not force:
        print("[*] CICIDS2017 dataset already downloaded.")
        return DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading CICIDS2017 dataset from HuggingFace...")
    for fname in CICIDS_FILES:
        dest = DATA_DIR / fname
        if dest.exists() and not force:
            print(f"    {fname} — already exists, skipping")
            continue
        url = f"{HF_BASE}/{fname.replace(' ', '%20')}?download=true"
        print(f"    Downloading {fname}...")
        subprocess.run(
            ["curl", "-L", "-o", str(dest), "--progress-bar", url],
            check=True,
        )

    marker.write_text("done")
    print("[+] Download complete.")
    return DATA_DIR


def load_cicids(sample_frac: float | None = None) -> pd.DataFrame:
    """Load all CICIDS2017 CSV files into a single DataFrame.

    Args:
        sample_frac: If set, sample this fraction of data (for faster iteration).

    Returns:
        DataFrame with cleaned columns and attack_category.
    """
    download_cicids()

    dfs = []
    for fname in CICIDS_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [!] Missing: {fname}, skipping")
            continue
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        dfs.append(df)
        print(f"    {fname}: {len(df)} records")

    df = pd.concat(dfs, ignore_index=True)

    # Strip whitespace from column names (CICIDS known issue)
    df.columns = df.columns.str.strip()

    # Clean Label column
    df["Label"] = df["Label"].astype(str).str.strip()

    # Normalize web attack labels with encoding issues (en-dash variants)
    df["Label"] = df["Label"].str.replace(
        r"Web Attack [^\w\s] ", "Web Attack \u2013 ", regex=True
    )

    # Map to attack categories
    df["attack_category"] = df["Label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")

    # Remove rows with unknown category
    n_unknown = (df["attack_category"] == "unknown").sum()
    if n_unknown > 0:
        print(f"  [!] Dropping {n_unknown} rows with unmapped labels")
        unmapped = df[df["attack_category"] == "unknown"]["Label"].unique()
        print(f"      Unmapped labels: {unmapped}")
        df = df[df["attack_category"] != "unknown"]

    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"  [*] Sampled to {len(df)} records ({sample_frac*100:.0f}%)")

    print(f"[+] Loaded CICIDS2017: {len(df)} records, "
          f"{df['attack_category'].nunique()} categories")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary statistics about the dataset."""
    return {
        "total_records": len(df),
        "features": len([c for c in df.columns if c not in ["Label", "attack_category"]]),
        "attack_categories": df["attack_category"].value_counts().to_dict(),
        "attack_types": df["Label"].nunique() if "Label" in df.columns else 0,
        "missing_values": df.isnull().sum().sum(),
    }
