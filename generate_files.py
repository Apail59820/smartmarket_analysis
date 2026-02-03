import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "data"
N_LEADS = 5000
N_CAMPAIGNS = 12
START_LEAD_ID = 10000

START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2025, 10, 1)

CHANNELS = ["Emailing", "Facebook Ads", "LinkedIn", "Instagram Ads"]
DEVICES = ["Mobile", "Desktop", "Tablet"]

COMPANY_SIZES = ["1-10", "10-50", "50-100", "100-500", "500-1000", "1000+"]
SECTORS = ["Tech", "Retail", "Finance", "Health", "Education", "Industry", "Energy", "Media", "Travel"]
REGIONS = ["IdF", "Hauts-de-France", "PAC", "ARA", "Occitanie", "Nouvelle-Aquitaine", "Bretagne", "Pays de la Loire", "Grand Est"]

STATUSES = ["MQL", "SQL", "Client"]

CHANNEL_WEIGHTS = [0.28, 0.30, 0.18, 0.24]
DEVICE_WEIGHTS = [0.62, 0.28, 0.10]

BASE_STATUS_PROBS = {"MQL": 0.60, "SQL": 0.28, "Client": 0.12}

STATUS_CHANNEL_BONUS = {
    "Emailing":      {"Client": 0.01, "SQL": 0.02, "MQL": -0.03},
    "Facebook Ads":  {"Client": 0.00, "SQL": 0.01, "MQL": -0.01},
    "LinkedIn":      {"Client": 0.03, "SQL": 0.04, "MQL": -0.07},
    "Instagram Ads": {"Client": 0.00, "SQL": 0.00, "MQL": 0.00},
}

SIZE_BONUS = {
    "1-10":     {"Client": -0.02, "SQL": -0.01, "MQL": 0.03},
    "10-50":    {"Client": -0.01, "SQL": 0.00,  "MQL": 0.01},
    "50-100":   {"Client": 0.00,  "SQL": 0.01,  "MQL": -0.01},
    "100-500":  {"Client": 0.01,  "SQL": 0.02,  "MQL": -0.03},
    "500-1000": {"Client": 0.02,  "SQL": 0.03,  "MQL": -0.05},
    "1000+":    {"Client": 0.03,  "SQL": 0.04,  "MQL": -0.07},
}

def clamp_probs(probs_dict):
    probs = np.array([max(0.0, probs_dict[s]) for s in STATUSES], dtype=float)
    if probs.sum() == 0:
        probs = np.array([1/3, 1/3, 1/3], dtype=float)
    probs = probs / probs.sum()
    return probs

def random_date(start, end):
    delta_days = (end - start).days
    day_offset = np.random.randint(0, delta_days)
    return start + timedelta(days=int(day_offset))

def generate_leads(n):
    lead_ids = np.arange(START_LEAD_ID, START_LEAD_ID + n, dtype=int)

    channels = np.random.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)
    devices = np.random.choice(DEVICES, size=n, p=DEVICE_WEIGHTS)
    dates = [random_date(START_DATE, END_DATE).date().isoformat() for _ in range(n)]

    leads_df = pd.DataFrame({
        "lead_id": lead_ids,
        "date": dates,
        "channel": channels,
        "device": devices
    })
    return leads_df

def generate_crm_from_leads(leads_df):
    n = len(leads_df)
    company_size = np.random.choice(COMPANY_SIZES, size=n, p=[0.28, 0.26, 0.16, 0.16, 0.08, 0.06])
    sector = np.random.choice(SECTORS, size=n)
    region = np.random.choice(REGIONS, size=n, p=[0.24, 0.10, 0.10, 0.12, 0.10, 0.10, 0.07, 0.08, 0.09])

    statuses = []
    for ch, cs in zip(leads_df["channel"].values, company_size):
        probs = BASE_STATUS_PROBS.copy()
        for s in STATUSES:
            probs[s] += STATUS_CHANNEL_BONUS.get(ch, {}).get(s, 0.0)
            probs[s] += SIZE_BONUS.get(cs, {}).get(s, 0.0)
        p = clamp_probs(probs)
        statuses.append(np.random.choice(STATUSES, p=p))

    crm_df = pd.DataFrame({
        "lead_id": leads_df["lead_id"].astype(int),
        "company_size": company_size,
        "sector": sector,
        "region": region,
        "status": statuses
    })
    return crm_df

def generate_campaigns(n_campaigns):
    campaigns = []
    for i in range(1, n_campaigns + 1):
        channel = CHANNELS[(i - 1) % len(CHANNELS)]
        campaign_id = f"CAMP{i:02d}"

        base_impr = np.random.randint(20000, 160000)

        ctr_mu = {
            "Emailing": 0.030,
            "Facebook Ads": 0.022,
            "LinkedIn": 0.018,
            "Instagram Ads": 0.028
        }[channel]
        ctr = np.clip(np.random.normal(ctr_mu, 0.005), 0.005, 0.08)

        clicks = int(round(base_impr * ctr))

        cvr_mu = {
            "Emailing": 0.070,
            "Facebook Ads": 0.055,
            "LinkedIn": 0.080,
            "Instagram Ads": 0.060
        }[channel]
        cvr = np.clip(np.random.normal(cvr_mu, 0.015), 0.01, 0.20)
        conversions = int(round(clicks * cvr))

        cpm_mu = {
            "Emailing": 18,
            "Facebook Ads": 35,
            "LinkedIn": 60,
            "Instagram Ads": 30
        }[channel]
        cpm = max(5, np.random.normal(cpm_mu, 8))
        cost = int(round((base_impr / 1000) * cpm))

        campaigns.append({
            "campaign_id": campaign_id,
            "channel": channel,
            "cost": cost,
            "impressions": int(base_impr),
            "clicks": int(clicks),
            "conversions": int(conversions)
        })
    return campaigns

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    leads_df = generate_leads(N_LEADS)
    leads_path = os.path.join(OUTPUT_DIR, "leads_smartmarket.csv")
    leads_df.to_csv(leads_path, index=False)

    crm_df = generate_crm_from_leads(leads_df)
    crm_path = os.path.join(OUTPUT_DIR, "crm_smartmarket.xlsx")
    crm_df.to_excel(crm_path, index=False)

    campaigns = generate_campaigns(N_CAMPAIGNS)
    camp_path = os.path.join(OUTPUT_DIR, "campaign_smartmarket.json")
    with open(camp_path, "w", encoding="utf-8") as f:
        json.dump(campaigns, f, ensure_ascii=False, indent=2)

    print("✅ Fichiers générés :")
    print(" -", leads_path, f"({len(leads_df)} lignes)")
    print(" -", crm_path, f"({len(crm_df)} lignes)")
    print(" -", camp_path, f"({len(campaigns)} campagnes)")
    print("\nAperçu leads:")
    print(leads_df.head())
    print("\nAperçu CRM:")
    print(crm_df.head())
    print("\nAperçu campaigns:")
    print(pd.DataFrame(campaigns).head())

if __name__ == "__main__":
    main()
