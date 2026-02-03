# Generate JSON file for the React DashBoard

import pandas as pd
import json

df = pd.read_csv("output/smartmarket_leads_scope.csv", parse_dates=["date"])
camp = pd.read_csv("output/smartmarket_campaigns_kpi.csv")

total_leads = int(df["lead_id"].nunique())

status_counts = df["status"].value_counts()
pct_client = float((status_counts.get("Client", 0) / total_leads) * 100)
pct_sql = float((status_counts.get("SQL", 0) / total_leads) * 100)

total_impr = camp["impressions"].sum()
total_clicks = camp["clicks"].sum()
total_conv = camp["conversions"].sum()
total_cost = camp["cost"].sum()

ctr = float(total_clicks / total_impr) if total_impr else None
cpl = float(total_cost / total_leads) if total_leads else None
cpa = float(total_cost / total_conv) if total_conv else None

leads_by_channel = (df.groupby("channel")["lead_id"].nunique()
                    .sort_values(ascending=False)
                    .reset_index(name="leads"))

status_by_channel = pd.crosstab(df["channel"], df["status"], normalize="index").reset_index()
for col in ["MQL","SQL","Client"]:
    if col not in status_by_channel.columns:
        status_by_channel[col] = 0.0

camp_channel = (camp.groupby("channel")
                .agg(cost=("cost","sum"))
                .reset_index()
                .merge(leads_by_channel, on="channel", how="left"))
camp_channel["cpl"] = camp_channel["cost"] / camp_channel["leads"]

data = {
  "kpi": {
    "totalLeads": total_leads,
    "pctClient": round(pct_client, 1),
    "pctSQL": round(pct_sql, 1),
    "ctr": round(ctr*100, 2) if ctr is not None else None,
    "cpl": round(cpl, 2) if cpl is not None else None,
    "cpa": round(cpa, 2) if cpa is not None else None,
  },
  "charts": {
    "leadsByChannel": leads_by_channel.to_dict(orient="records"),
    "statusByChannel": status_by_channel.to_dict(orient="records"),
    "cplByChannel": camp_channel[["channel","cpl"]].sort_values("cpl", ascending=False).to_dict(orient="records")
  }
}

with open("output/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("OK -> dashboard_data.json généré")
