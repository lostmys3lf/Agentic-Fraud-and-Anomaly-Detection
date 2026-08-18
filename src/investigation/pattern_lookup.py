"""
L2 Investigation -- historical pattern lookup over the event tables.

The other half of L2. sop_retriever.py answers "what does policy say?"; this module
answers "what has this customer actually done?". L3 needs both: a policy clause with no
evidence decides nothing, and evidence with no clause cannot be cited.

Scope note (CLAUDE.md data gaps -- do not invent values for these):
  - There is no case_id linking transactions, swap events and complaints, so every
    lookup here is keyed on customer_id and nothing else.
  - SOP-002's "same e-wallet id or card number" does not exist in the data.
    payment_method is a category only; device_id is the one usable shared fingerprint.
  - SOP-003's "cancel and re-subscribe within 7 days" and "no usage between redemptions"
    are not computable -- there is no subscription or usage table. Count redemptions and
    say plainly that the cancel-recreate half of the rule is unverified.
  - complaint_text is 47 repeated templates across 280 rows. Match it as categories,
    not as free text, and treat related_to_fraud_case as a proxy label.

Timestamps arrive as strings from read_csv unless parsed -- parse before comparing.
"""

import pandas as pd

import config


def find_shared_device_accounts(
    transactions: pd.DataFrame,
    device_id: str,
    window_hours: int = config.SHARED_DEVICE_WINDOW_HOURS,
) -> pd.DataFrame:
 
    df = transactions[transactions["device_id"] == device_id].copy()

    # Device yang nggak pernah dipakai bukan error, cuma nggak ada temuan.
    # Bentuk kolomnya tetap lengkap supaya pemanggil nggak perlu pengecekan tambahan.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "n_transactions",
                "first_seen",
                "last_seen",
                "max_accounts_in_window",
            ]
        )

    # read_csv ngasih timestamp sebagai teks. Dibandingin sebagai teks,
    # '2026-05-10' < '2026-05-9' -- benar secara abjad, salah secara waktu.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Tiap transaksi jadi titik awal jendelanya sendiri. SOP-002 ngitung "dalam 24 jam",
    # bukan "sepanjang data": 25 akun selama 3 bulan itu warnet, 25 akun dalam semalam
    # itu temuan lain.
    accounts_per_window = []
    for start in df["timestamp"]:
        end = start + pd.Timedelta(hours=window_hours)
        window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        accounts_per_window.append(window["customer_id"].nunique())
    max_accounts = max(accounts_per_window)

    # Angka puncak di atas dipakai L3 buat mutusin; tabel per akun ini yang dikutip L4.
    per_account = (
        df.groupby("customer_id")
        .agg(
            n_transactions=("transaction_id", "count"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )
    per_account["max_accounts_in_window"] = max_accounts

    return per_account.sort_values("n_transactions", ascending=False).reset_index(drop=True)


def count_promo_redemptions(
    transactions: pd.DataFrame,
    customer_id: str,
    window_days: int = config.PROMO_WINDOW_DAYS,
) -> int:
 
    df = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"] == "promo_bundle_redeem")
    ].copy()

    if df.empty:
        return 0

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Data sintetis nggak punya "hari ini". Acuannya penebusan terakhir nasabah ini
    # sendiri, jadi jendelanya selalu berakhir di aktivitas terakhir dia -- bukan di
    # tanggal kalender yang sama buat semua orang. Datanya cuma 90 hari, jadi dengan
    # PROMO_WINDOW_DAYS=90 saringan ini nggak buang baris apa pun (8 -> 8 pada
    # CUST10061). Kodenya bener, datanya yang bikin langkah ini nggak berefek.
    cutoff = df["timestamp"].max() - pd.Timedelta(days=window_days)

    # int() supaya hasilnya bisa masuk json.dumps di laporan L4 -- numpy.int64 nggak bisa.
    return int(len(df[df["timestamp"] >= cutoff]))


def get_customer_complaints(
    complaint_notes: pd.DataFrame,
    customer_id: str,
) -> pd.DataFrame:
  
    df = complaint_notes[complaint_notes["customer_id"] == customer_id].copy()

    # Diurutin sebagai waktu, bukan teks. Di data ini formatnya ISO jadi urutan abjad
    # kebetulan sama, tapi itu sifat formatnya -- bukan sesuatu yang boleh diandalkan.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.sort_values("timestamp")[
        [
            "complaint_id",
            "timestamp",
            "channel",
            "complaint_text",
            "related_to_fraud_case",
        ]
    ].reset_index(drop=True)


def build_customer_profile(
    customer_id: str,
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    sim_swap_events: pd.DataFrame,
    complaint_notes: pd.DataFrame,
) -> dict:
  
    row = customers[customers["customer_id"] == customer_id]

    # Beda sama count_promo_redemptions yang balikin 0 buat id nggak dikenal. Di sana 0
    # itu jawaban sah ("nol penebusan"); di sini profil kosong bukan jawaban, itu tanda
    # pemanggilnya salah. Kalau dibiarin lolos, L3 nerima profil serba nol dan mutusin
    # Auto-Approve dengan tenang.
    if row.empty:
        raise KeyError(f"unknown customer_id: {customer_id}")
    row = row.iloc[0]

    df = transactions[transactions["customer_id"] == customer_id].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    hour = df["timestamp"].dt.hour
    night = df[(hour >= config.NIGHT_HOUR_START) & (hour < config.NIGHT_HOUR_END)]

    # Satu nasabah bisa pakai beberapa device, dan find_shared_device_accounts cuma
    # nerima satu -- itu yang bikin loop ini perlu. Angkanya diambil dari fungsi itu,
    # bukan dihitung ulang, supaya definisi "dalam 24 jam" cuma hidup di satu tempat.
    shared_device_ids = []
    accounts_per_device = []
    for device_id in df["device_id"].unique():
        shared = find_shared_device_accounts(transactions, device_id)
        n_accounts = int(shared["max_accounts_in_window"].iloc[0])
        if n_accounts > 1:  # dipakai satu akun doang bukan temuan
            shared_device_ids.append(device_id)
            accounts_per_device.append(n_accounts)

    swaps = sim_swap_events[sim_swap_events["customer_id"] == customer_id]
    complaints = get_customer_complaints(complaint_notes, customer_id)

    # Semua angka dibungkus int()/float() dan semua daftar pakai .tolist(): tipe bawaan
    # pandas (numpy.int64, numpy array) nggak bisa masuk json.dumps di laporan L4.
    return {
        "customer_id": customer_id,
        "home_city": row["home_city"],
        "tenure_months": int(row["tenure_months"]),
        "device_changes_last_12mo": int(row["device_changes_last_12mo"]),
        "n_transactions": len(df),
        "n_night_transactions": len(night),
        "promo_redemptions_90d": count_promo_redemptions(transactions, customer_id),
        "shared_device_ids": shared_device_ids,
        "max_accounts_per_shared_device": max(accounts_per_device, default=0),
        "n_sim_swaps": len(swaps),
        # Nasabah tanpa swap wajar, bukan error -- .max() atas kolom kosong itu NaN,
        # dan NaN di JSON jadi literal `NaN` yang bukan JSON sah.
        "max_swap_distance_km": (
            float(swaps["distance_from_home_km"].max()) if not swaps.empty else 0.0
        ),
        # Dua key di bawah dipakai indikator 2 dan 3 SOP-001 (L3 evaluate_sim_swap).
        "swap_reasons_stated": swaps["reason_stated"].tolist(),
        # Sengaja None, bukan 0.0 -- beda dari max_swap_distance_km di atas. Jarak 0 km itu
        # kasus paling aman (nasabah di rumah), tapi 0 jam artinya "login baru diganti
        # beberapa detik lalu", yaitu kasus paling PARAH. Nasabah tanpa swap bakal ketuduh
        # indikator yang nggak pernah kejadian. None = "nggak ada nilainya", dan
        # evaluate_sim_swap() keluar duluan buat nasabah tanpa swap jadi nilai ini nggak
        # pernah dibandingin. None juga jadi null yang sah di JSON, beda dari float('inf').
        "min_hours_since_login_change": (
            float(swaps["hours_since_last_login_change"].min()) if not swaps.empty else None
        ),
        "complaint_ids": complaints["complaint_id"].tolist(),
        "transaction_ids": df["transaction_id"].tolist(),
        "swap_event_ids": swaps["event_id"].tolist(),
    }
