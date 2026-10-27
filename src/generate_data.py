"""Generate synthetic free-text feedback from informal-market outlet owners.

Each comment is assembled from phrase fragments for one underlying theme, with
noise words, typos and mixed phrasing so the clustering has to work for it.
The hidden `true_theme` column is kept only to *evaluate* the unsupervised model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data"

THEMES = {
    "Network quality": [
        "the signal is very weak inside the shop", "network keeps dropping in the afternoon",
        "customers complain data is slow here", "no signal when it rains", "the speed test was very slow",
        "calls drop all the time", "network coverage is bad in this area", "slow internet for customers",
    ],
    "SIM & stock supply": [
        "we ran out of sim cards last week", "the rep does not bring enough stock",
        "starter packs finish quickly", "need more sim stock every month", "stock delivery is late",
        "out of stock of sims for two weeks", "supplier only comes once a month with stock",
    ],
    "Commission & payment": [
        "commission is paid late", "the commission is too small for the work",
        "i do not understand how commission is calculated", "payment for sims sold came late",
        "we want better commission on airtime", "commission statement is confusing",
    ],
    "Terminal & device issues": [
        "the airtime machine is offline often", "terminal battery does not last",
        "the device prints slips badly", "machine needs to be fixed", "terminal froze and lost a sale",
        "the printer paper runs out on the device", "terminal software update failed",
    ],
    "Branding & visibility": [
        "we would like a new board outside", "please paint the wall with branding",
        "the old poster is faded", "customers do not know we sell sims", "give us branded umbrella and banner",
        "no signage outside the shop", "need branding so people see us from the road",
    ],
    "Training & support": [
        "nobody showed me how to register a sim", "need training on rica registration",
        "the helpline does not answer", "rep never visits to help", "i need help with the app",
        "support takes days to respond", "want training for my assistant",
    ],
}
FILLERS = ["", "", "", "please help.", "thank you.", "this is a big problem.", "eish.", "as discussed before.",
           "it has been like this for months.", "we are losing customers."]
TOWNS = ["Komani", "Whittlesea", "Cofimvaba", "Lady Frere", "Tarkastad", "Sterkstroom"]
# themes are not evenly spread across towns: rural towns skew to network & stock issues
TOWN_THEME_BIAS = {"Komani": [1, 1, 1.4, 1.2, 1.3, 1], "Whittlesea": [1.6, 1.3, 1, 1, 1, 1],
                   "Cofimvaba": [2.0, 1.5, 1, 1, .8, 1.2], "Lady Frere": [2.2, 1.6, 1, 1, .8, 1.3],
                   "Tarkastad": [1.5, 1.4, 1, 1.1, 1, 1], "Sterkstroom": [1.4, 1.2, 1, 1.2, 1, 1]}


def typo(s):
    if RNG.random() < 0.15 and len(s) > 10:
        i = RNG.integers(1, len(s) - 1)
        s = s[:i] + s[i + 1:]
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, themes = [], list(THEMES)
    for i in range(640):
        town = RNG.choice(TOWNS, p=[.36, .16, .15, .14, .1, .09])
        w = np.array(TOWN_THEME_BIAS[town]); theme = RNG.choice(themes, p=w / w.sum())
        k = RNG.integers(1, 3)
        parts = list(RNG.choice(THEMES[theme], k, replace=False))
        # 20% of comments mention a second theme in passing
        if RNG.random() < 0.2:
            parts.append(RNG.choice(THEMES[RNG.choice([t for t in themes if t != theme])]))
        text = " and ".join(parts) + ". " + RNG.choice(FILLERS)
        text = typo(text.strip().capitalize())
        rows.append({"response_id": f"R{i + 1:04d}", "town": town, "comment": text, "true_theme": theme})
    pd.DataFrame(rows).to_csv(OUT / "outlet_feedback.csv", index=False)
    print(f"wrote {len(rows)} comments -> {OUT / 'outlet_feedback.csv'}")


if __name__ == "__main__":
    main()
