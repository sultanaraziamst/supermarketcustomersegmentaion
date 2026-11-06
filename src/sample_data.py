"""Synthetic supermarket transactions with the SAME schema as the UCI
'Online Retail' dataset:

    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

Six latent shopper archetypes are simulated (champions, loyal, big spenders,
new, at-risk, hibernating) and realistic data-quality problems are injected
(cancellations, guest checkouts, duplicates, zero prices, postage lines) so the
cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from .config import RANDOM_SEED, SAMPLE_FILE

CATALOG = {
    "Fruit & Veg": [("BANANAS 1KG", 1.2), ("RED APPLES 6PK", 2.4), ("BROCCOLI HEAD", 0.9), ("CARROTS 1KG", 0.8),
                    ("TOMATOES VINE 500G", 1.9), ("POTATOES 2.5KG", 2.1), ("AVOCADO RIPE 2PK", 1.8), ("SPINACH BAG", 1.3)],
    "Dairy": [("WHOLE MILK 2L", 1.6), ("CHEDDAR BLOCK 400G", 3.2), ("GREEK YOGHURT 500G", 1.7), ("BUTTER SALTED 250G", 2.3),
              ("FREE RANGE EGGS 12", 2.9), ("DOUBLE CREAM 300ML", 1.4), ("MOZZARELLA BALL", 1.1), ("SEMI SKIMMED MILK 4PT", 1.5)],
    "Bakery": [("WHITE SLICED LOAF", 1.1), ("SOURDOUGH BATARD", 2.6), ("CROISSANTS 4PK", 1.8), ("BAGELS PLAIN 5PK", 1.5),
               ("WHOLEMEAL ROLLS 6PK", 1.2), ("CHOCOLATE MUFFINS 4PK", 2.0), ("PITTA BREAD 6PK", 0.9)],
    "Meat & Fish": [("CHICKEN BREAST 1KG", 6.5), ("MINCED BEEF 500G", 4.2), ("SALMON FILLETS 2PK", 5.5), ("BACON STREAKY 300G", 3.1),
                    ("PORK SAUSAGES 8PK", 2.8), ("PRAWNS KING 200G", 4.8), ("LAMB CHOPS 4PK", 7.2)],
    "Beverages": [("ORANGE JUICE 1L", 1.9), ("COLA 6X330ML", 3.0), ("SPARKLING WATER 6X1.5L", 3.4), ("GROUND COFFEE 227G", 3.9),
                  ("BLACK TEA 80 BAGS", 2.5), ("APPLE SQUASH 1L", 1.3), ("ENERGY DRINK 4PK", 3.6)],
    "Snacks": [("SALTED CRISPS 6PK", 1.9), ("MILK CHOCOLATE BAR", 1.0), ("MIXED NUTS 300G", 3.3), ("DIGESTIVE BISCUITS", 1.2),
               ("POPCORN SWEET 5PK", 1.6), ("GUMMY BEARS 180G", 0.9), ("CEREAL BARS 6PK", 2.1)],
    "Household": [("WASHING LIQUID 1.5L", 5.5), ("KITCHEN ROLL 4PK", 3.2), ("DISHWASHER TABS 40", 6.2), ("BIN LINERS 30PK", 2.2),
                  ("TOILET TISSUE 9 ROLL", 4.8), ("MULTI SURFACE SPRAY", 1.7), ("SPONGE SCOURERS 6PK", 1.4)],
    "Personal Care": [("SHAMPOO 400ML", 3.1), ("TOOTHPASTE 100ML", 1.8), ("SHOWER GEL 250ML", 1.6), ("DEODORANT SPRAY", 2.4),
                      ("HAND WASH 500ML", 1.5), ("COTTON PADS 100", 1.1)],
    "Frozen": [("FROZEN PEAS 1KG", 1.7), ("MARGHERITA PIZZA", 2.5), ("VANILLA ICE CREAM 2L", 3.4), ("FISH FINGERS 10PK", 2.6),
               ("OVEN CHIPS 1.5KG", 2.2), ("FROZEN BERRIES 500G", 3.0)],
    "Pantry": [("SPAGHETTI 500G", 0.9), ("BASMATI RICE 1KG", 2.4), ("CHOPPED TOMATOES 400G", 0.6), ("OLIVE OIL 500ML", 5.2),
               ("PLAIN FLOUR 1.5KG", 1.2), ("BAKED BEANS 4PK", 2.0), ("PEANUT BUTTER 340G", 2.1), ("TOMATO KETCHUP 460G", 1.9)],
}

# name, share, invoices ~ Poisson(lam), first-purchase day range, last-purchase day range,
# mean lines/invoice, mean units/line, price multiplier, return probability
ARCHETYPES = [
    ("champions",    0.12, 30.0, (0, 120),   (335, 364), 14, 2.5, 1.0, 0.03),
    ("loyal",        0.22, 14.0, (0, 150),   (290, 364), 9,  2.0, 1.0, 0.02),
    ("big_spenders", 0.05, 7.0,  (0, 200),   (250, 364), 30, 12.0, 0.8, 0.02),
    ("new",          0.18, 1.5,  (300, 355), (330, 364), 7,  1.5, 1.0, 0.01),
    ("at_risk",      0.20, 8.0,  (0, 120),   (110, 250), 8,  2.0, 1.0, 0.03),
    ("hibernating",  0.23, 1.5,  (0, 200),   (30, 200),  6,  1.5, 1.0, 0.02),
]
COUNTRIES = ["United Kingdom", "Ireland", "France", "Germany", "Spain", "Netherlands", "Portugal"]
COUNTRY_P = [0.88, 0.03, 0.025, 0.025, 0.015, 0.015, 0.01]
HOUR_P = np.array([2, 5, 9, 12, 12, 9, 7, 7, 8, 9, 6, 3, 1], dtype=float)  # hours 8..20
HOUR_P /= HOUR_P.sum()


def generate(n_customers: int = 1500, seed: int = RANDOM_SEED, start: str = "2024-01-01",
             days: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)

    products = [(f"{10000 + i}", name, price, cat)
                for i, (name, price, cat) in enumerate(
                    (n, p, c) for c, items in CATALOG.items() for n, p in items)]
    cats = list(CATALOG)
    cat_products = {c: [i for i, p in enumerate(products) if p[3] == c] for c in cats}

    shares = np.array([a[1] for a in ARCHETYPES])
    cust_ids = rng.choice(np.arange(12000, 12000 + n_customers * 3), n_customers, replace=False)
    invoices = []  # (timestamp, customer_id, country, lines, is_return)

    for cid in cust_ids:
        name, _, lam, f_rng, l_rng, lines_mean, qty_mean, pmult, p_ret = ARCHETYPES[
            rng.choice(len(ARCHETYPES), p=shares / shares.sum())]
        first = int(rng.integers(f_rng[0], f_rng[1] + 1))
        last = int(rng.integers(max(first, l_rng[0]), max(first, l_rng[0], l_rng[1]) + 1))
        last = min(last, days - 1)
        n_inv = max(1, int(rng.poisson(lam)))
        d = np.sort(rng.integers(first, last + 1, n_inv))
        d[-1] = last
        if n_inv > 1:
            d[0] = first
        country = rng.choice(COUNTRIES, p=COUNTRY_P)
        cat_pref = rng.dirichlet(np.ones(len(cats)) * 0.8)

        for day in d:
            hour = int(rng.choice(np.arange(8, 21), p=HOUR_P))
            ts = start_ts + pd.Timedelta(days=int(day), hours=hour, minutes=int(rng.integers(0, 60)))
            n_lines = 1 + rng.poisson(lines_mean)
            lines = []
            for c_i in rng.choice(len(cats), n_lines, p=cat_pref):
                pid = rng.choice(cat_products[cats[c_i]])
                code, desc, price, _ = products[pid]
                qty = 1 + rng.poisson(qty_mean - 1 if qty_mean > 1 else 0.2)
                lines.append([code, desc, int(qty), round(price * pmult * rng.lognormal(0, 0.04), 2)])
            if rng.random() < 0.05:
                lines.append(["POST", "POSTAGE", 1, 12.0])
            invoices.append((ts, cid, country, lines, False))
            if rng.random() < p_ret:
                k = max(1, len(lines) // 4)
                ret = [[l[0], l[1], -l[2], l[3]] for l in rng.permutation(np.array(lines, dtype=object))[:k].tolist()]
                invoices.append((ts + pd.Timedelta(days=int(rng.integers(1, 6))), cid, country, ret, True))

    # guest checkouts (no CustomerID)
    for _ in range(int(0.03 * len(invoices))):
        ts = start_ts + pd.Timedelta(days=int(rng.integers(0, days)), hours=int(rng.integers(8, 21)))
        lines = []
        for _ in range(1 + rng.poisson(5)):
            code, desc, price, _ = products[rng.integers(0, len(products))]
            lines.append([code, desc, 1 + int(rng.poisson(1)), price])
        invoices.append((ts, None, "United Kingdom", lines, False))

    invoices.sort(key=lambda x: x[0])
    rows = []
    for no, (ts, cid, country, lines, is_ret) in enumerate(invoices, start=540000):
        inv = f"C{no}" if is_ret else str(no)
        for code, desc, qty, price in lines:
            rows.append((inv, code, desc, qty, ts.strftime("%Y-%m-%d %H:%M:%S"), price,
                         "" if cid is None else str(cid), country))
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])

    # inject dirt
    dup = df.sample(frac=0.003, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    zero_idx = df.sample(frac=0.002, random_state=seed + 1).index
    df.loc[zero_idx, "UnitPrice"] = 0.0
    null_desc = df.sample(frac=0.002, random_state=seed + 2).index
    df.loc[null_desc, "Description"] = None
    return df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)


def write_sample(path=SAMPLE_FILE, **kwargs) -> str:
    df = generate(**kwargs)
    df.to_csv(path, index=False)
    print(f"[sample] wrote {len(df):,} rows -> {path}")
    return str(path)


if __name__ == "__main__":
    write_sample()
