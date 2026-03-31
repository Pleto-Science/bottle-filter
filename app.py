import io
from datetime import datetime

import pandas as pd
import streamlit as st

# ==========================================================
# Config
# ==========================================================

st.set_page_config(
    page_title="Bottle Filter",
    page_icon="🍶",
    layout="centered",
)

# ==========================================================
# Shared helpers
# ==========================================================

def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".0", "", regex=False).str.strip()


ORDER_COL_CANDIDATES = ["order_id", "id", "name", "order_number", "external_order_id"]
DATE_COL_CANDIDATES  = ["created_at", "processed_at", "day", "scheduled_at"]
SKU_COL_CANDIDATES   = ["line_item_sku", "sku", "variant_sku"]
CUST_COL_CANDIDATES  = ["customer_id", "customer", "client_id"]


def detect_col(df: pd.DataFrame, candidates: list, label: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calculate_strict_gap(group: pd.DataFrame, order_col: str, sku_col: str, sku_target: str) -> int:
    all_orders    = group[order_col].unique().tolist()
    bottle_orders = group[group[sku_col] == sku_target][order_col].unique().tolist()
    if not bottle_orders:
        return -1
    last_idx = all_orders.index(bottle_orders[-1])
    return len(all_orders[last_idx + 1:])


def render_results(result_df: pd.DataFrame, filename_prefix: str, gap_label: str = None):
    """Shared results block: metrics strip, table, download button."""
    if result_df is None:
        return
    if result_df.empty:
        st.warning("No matching customers found.")
        return
    st.success(f"{len(result_df)} charge rows ready for export.")
    st.dataframe(result_df, use_container_width=True)
    csv_bytes = result_df.to_csv(index=False).encode()
    filename  = f"{filename_prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


# ==========================================================
# Second Bottle processing
# ==========================================================

@st.cache_data(show_spinner=False)
def process_second_bottle(orders_bytes: bytes, charges_bytes: bytes, sku_target: str, gap_threshold: int):
    errors = []

    orders_df = pd.read_csv(io.BytesIO(orders_bytes))
    orders_df.columns = orders_df.columns.str.strip().str.lower()

    cust_col  = detect_col(orders_df, CUST_COL_CANDIDATES, "customer_id")
    order_col = detect_col(orders_df, ORDER_COL_CANDIDATES, "order_id")
    sku_col   = detect_col(orders_df, SKU_COL_CANDIDATES, "sku")
    date_col  = detect_col(orders_df, DATE_COL_CANDIDATES, "date")

    missing = [lbl for col, lbl in [(cust_col, "customer_id"), (order_col, "order_id"), (sku_col, "sku")] if col is None]
    if missing:
        errors.append(f"Orders CSV missing columns for: **{', '.join(missing)}**. Found: `{list(orders_df.columns)}`")
        return None, None, errors

    orders_df[cust_col]  = orders_df[cust_col].ffill()
    orders_df[order_col] = orders_df[order_col].ffill()
    orders_df[cust_col]  = normalize_id(orders_df[cust_col])

    if date_col:
        orders_df[date_col] = pd.to_datetime(orders_df[date_col], errors="coerce")
        orders_df = orders_df.sort_values([cust_col, date_col]).reset_index(drop=True)
    else:
        orders_df = orders_df.sort_values(cust_col).reset_index(drop=True)

    gap_series = orders_df.groupby(cust_col).apply(
        calculate_strict_gap,
        order_col=order_col,
        sku_col=sku_col,
        sku_target=sku_target,
        include_groups=False,
    )
    gap_df = gap_series.reset_index()
    gap_df.columns = [cust_col, "orders_since_last_bottle"]
    qualifying = set(gap_df[gap_df["orders_since_last_bottle"] >= gap_threshold][cust_col])

    charges_df = pd.read_csv(io.BytesIO(charges_bytes))
    charges_df.columns = charges_df.columns.str.strip().str.lower()

    if "customer_id" not in charges_df.columns:
        errors.append(f"Charges CSV missing `customer_id`. Found: `{list(charges_df.columns)}`")
        return None, None, errors

    charges_df["customer_id"] = normalize_id(charges_df["customer_id"])

    if "email" not in charges_df.columns and "email" in orders_df.columns:
        email_map = (
            orders_df[[cust_col, "email"]].ffill()
            .drop_duplicates(subset=cust_col)
            .rename(columns={cust_col: "customer_id"})
        )
        charges_df = charges_df.merge(email_map, on="customer_id", how="left")

    result_df = (
        charges_df[charges_df["customer_id"].isin(qualifying)]
        .drop_duplicates(subset=["customer_id"], keep="first")
        .copy()
    )

    export_cols = [c for c in ["charge_id", "customer_id", "email", "scheduled_at", "total_price"] if c in result_df.columns]
    stats = {
        "total_customers":    orders_df[cust_col].nunique(),
        "customers_with_sku": int((gap_df["orders_since_last_bottle"] >= 0).sum()),
        "qualifying":         len(qualifying),
        "charges_matched":    len(result_df),
        "date_col_found":     date_col is not None,
    }
    return result_df[export_cols], stats, errors


# ==========================================================
# First Bottle processing
# ==========================================================

@st.cache_data(show_spinner=False)
def process_first_bottle(orders_bytes: bytes, charges_bytes: bytes, customers_bytes: bytes,
                          sku_target: str, streak_threshold: int):
    errors = []

    # --- Orders: identify who already got a bottle ---
    orders_df = pd.read_csv(io.BytesIO(orders_bytes))
    orders_df.columns = orders_df.columns.str.strip().str.lower()

    cust_col = detect_col(orders_df, CUST_COL_CANDIDATES, "customer_id")
    sku_col  = detect_col(orders_df, SKU_COL_CANDIDATES, "sku")

    missing = [lbl for col, lbl in [(cust_col, "customer_id"), (sku_col, "sku")] if col is None]
    if missing:
        errors.append(f"Orders CSV missing columns for: **{', '.join(missing)}**. Found: `{list(orders_df.columns)}`")
        return None, None, errors

    orders_df[cust_col] = orders_df[cust_col].ffill()
    if "email" in orders_df.columns:
        orders_df["email"] = orders_df["email"].ffill()
    orders_df[cust_col] = normalize_id(orders_df[cust_col])

    botella_customers = set(
        orders_df[orders_df[sku_col] == sku_target][cust_col].unique()
    )
    total_customers = orders_df[cust_col].nunique()

    # --- Charges: remove existing bottle customers ---
    charges_df = pd.read_csv(io.BytesIO(charges_bytes))
    charges_df.columns = charges_df.columns.str.strip().str.lower()

    if "customer_id" not in charges_df.columns:
        errors.append(f"Charges CSV missing `customer_id`. Found: `{list(charges_df.columns)}`")
        return None, None, errors

    charges_df["customer_id"] = normalize_id(charges_df["customer_id"])
    original_count = len(charges_df)
    charges_df = charges_df[~charges_df["customer_id"].isin(botella_customers)].copy()
    removed_count = original_count - len(charges_df)

    # --- Customers: filter by streak ---
    customers_df = pd.read_csv(io.BytesIO(customers_bytes))
    customers_df.columns = customers_df.columns.str.strip().str.lower()

    cust_col2 = detect_col(customers_df, CUST_COL_CANDIDATES, "customer_id")
    if cust_col2 is None:
        errors.append(f"Customers CSV missing `customer_id`. Found: `{list(customers_df.columns)}`")
        return None, None, errors

    if "streak_of_uncancelled_charges" not in customers_df.columns:
        errors.append(
            f"Customers CSV missing `streak_of_uncancelled_charges`. Found: `{list(customers_df.columns)}`"
        )
        return None, None, errors

    customers_df[cust_col2] = normalize_id(customers_df[cust_col2])
    customers_df["streak_of_uncancelled_charges"] = pd.to_numeric(
        customers_df["streak_of_uncancelled_charges"], errors="coerce"
    )
    streak_customers = set(
        customers_df[customers_df["streak_of_uncancelled_charges"] >= streak_threshold][cust_col2].dropna()
    )

    # --- Final match ---
    result_df = charges_df[charges_df["customer_id"].isin(streak_customers)].copy()

    if "charge_id" not in result_df.columns:
        errors.append(f"Charges CSV missing `charge_id`. Found: `{list(charges_df.columns)}`")
        return None, None, errors

    # Email fallback from orders
    if "email" not in result_df.columns and "email" in orders_df.columns:
        email_map = (
            orders_df[[cust_col, "email"]].drop_duplicates(subset=cust_col)
            .rename(columns={cust_col: "customer_id"})
        )
        result_df = result_df.merge(email_map, on="customer_id", how="left")

    export_cols = [c for c in ["charge_id", "customer_id", "email"] if c in result_df.columns]
    result_df = result_df[export_cols].drop_duplicates()

    stats = {
        "total_customers":    total_customers,
        "botella_customers":  len(botella_customers),
        "streak_customers":   len(streak_customers),
        "removed_charges":    removed_count,
        "charges_matched":    len(result_df),
    }
    return result_df, stats, errors


# ==========================================================
# UI
# ==========================================================

st.title("🍶 Bottle Filter")

tab1, tab2 = st.tabs(["🔁 Second Bottle", "🆕 First Bottle"])

# ----------------------------------------------------------
# TAB 1 — Second Bottle
# ----------------------------------------------------------
with tab1:
    st.subheader("Second Bottle")
    st.caption("Customers who bought the SKU but haven't reordered in N+ orders.")

    with st.sidebar:
        st.header("Second Bottle Settings")
        sku_target    = st.text_input("Target SKU", value="BOTELLA-NEW-BRAND", key="sku_2nd")
        gap_threshold = st.number_input("Min orders since last bottle", min_value=1, value=3, step=1, key="gap_2nd")
        st.divider()
        st.markdown("**Files needed**")
        st.markdown("1. Orders CSV (full history with SKUs)\n2. Charges CSV (queued / upcoming)")

    col1, col2 = st.columns(2)
    with col1:
        orders_file_2nd = st.file_uploader("Orders CSV", type="csv", key="orders_2nd",
                                            help="Full order history with SKUs")
    with col2:
        charges_file_2nd = st.file_uploader("Charges CSV", type="csv", key="charges_2nd",
                                             help="Upcoming / queued charges")

    if orders_file_2nd and charges_file_2nd:
        with st.spinner("Processing…"):
            result_df, stats, errors = process_second_bottle(
                orders_file_2nd.getvalue(),
                charges_file_2nd.getvalue(),
                sku_target,
                gap_threshold,
            )

        if errors:
            for err in errors:
                st.error(err)
        else:
            if not stats["date_col_found"]:
                st.warning("No date column found — order chronology relies on CSV row order.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total customers",      stats["total_customers"])
            m2.metric("Ever bought SKU",      stats["customers_with_sku"])
            m3.metric(f"Gap ≥ {gap_threshold}", stats["qualifying"])
            m4.metric("Charges matched",      stats["charges_matched"])

            st.divider()
            render_results(result_df, "second_bottle")
    else:
        st.info("Upload both files above to get started.")


# ----------------------------------------------------------
# TAB 2 — First Bottle
# ----------------------------------------------------------
with tab2:
    st.subheader("First Bottle")
    st.caption("Customers who never received the SKU but have a streak ≥ N uncancelled charges.")

    streak_threshold = st.number_input("Min streak of uncancelled charges", min_value=1, value=3, step=1, key="streak_1st")
    sku_target_1st   = st.text_input("Target SKU", value="BOTELLA-NEW-BRAND", key="sku_1st")

    col1, col2, col3 = st.columns(3)
    with col1:
        orders_file_1st = st.file_uploader("Orders CSV", type="csv", key="orders_1st",
                                            help="Full order history — used to detect who already got the bottle")
    with col2:
        charges_file_1st = st.file_uploader("Charges CSV", type="csv", key="charges_1st",
                                             help="Upcoming / queued charges")
    with col3:
        customers_file_1st = st.file_uploader("Customers CSV", type="csv", key="customers_1st",
                                               help="Must contain streak_of_uncancelled_charges column")

    if orders_file_1st and charges_file_1st and customers_file_1st:
        with st.spinner("Processing…"):
            result_df, stats, errors = process_first_bottle(
                orders_file_1st.getvalue(),
                charges_file_1st.getvalue(),
                customers_file_1st.getvalue(),
                sku_target_1st,
                streak_threshold,
            )

        if errors:
            for err in errors:
                st.error(err)
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total customers",       stats["total_customers"])
            m2.metric("Already got bottle",    stats["botella_customers"])
            m3.metric("Charges removed",       stats["removed_charges"])
            m4.metric(f"Streak ≥ {streak_threshold}", stats["streak_customers"])
            m5.metric("Charges matched",       stats["charges_matched"])

            st.divider()
            render_results(result_df, "first_bottle")
    else:
        st.info("Upload all three files above to get started.")
