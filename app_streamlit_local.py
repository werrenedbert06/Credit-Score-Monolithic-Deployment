import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("RF-5_pipeline.pkl")

# Header
st.title("💳 Credit Score Predictor")
st.caption("Credit Classification: **Poor**, **Standard**, atau **Good**")
st.divider()

# Form
with st.form("credit_form"):

    st.subheader("👤 Profil Pribadi")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=30)
    with c2:
        month = st.selectbox("Bulan", ["Januari","Februari","Maret","April","Mei","Juni",
                                       "Juli","Agustus","September","Oktober","November","Desember"])
    with c3:
        occupation = st.selectbox("Pekerjaan", ["Akuntan","Arsitek","Developer","Dokter","Insinyur",
                                                "Pengusaha","Jurnalis","Pengacara","Manajer","Mekanik",
                                                "Manajer Media","Musisi","Ilmuwan","Guru","Penulis"])

    st.subheader("💰 Pendapatan & Rekening")
    c1, c2 = st.columns(2)
    with c1:
        annual_income           = st.number_input("Pendapatan Tahunan (USD)", min_value=0, value=50000, step=1000)
        monthly_inhand_salary   = st.number_input("Gaji Bersih Bulanan (USD)", min_value=0.0, value=3000.0, step=100.0)
        monthly_balance         = st.number_input("Saldo Bulanan (USD)", min_value=0.0, value=500.0, step=10.0)
        amount_invested_monthly = st.number_input("Investasi Bulanan (USD)", min_value=0.0, value=200.0, step=10.0)
    with c2:
        num_bank_accounts        = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=20, value=3)
        num_credit_card          = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=20, value=4)
        total_emi_per_month      = st.number_input("Total Angsuran per Bulan (USD)", min_value=0.0, value=100.0, step=10.0)
        credit_history_age_months = st.number_input("Usia Riwayat Kredit (bulan)", min_value=0, value=120)

    st.subheader("🏦 Utang & Risiko Kredit")
    c1, c2 = st.columns(2)
    with c1:
        outstanding_debt         = st.number_input("Sisa Utang (USD)", min_value=0.0, value=1000.0, step=100.0)
        credit_utilization_ratio = st.number_input("Rasio Penggunaan Kartu Kredit (%)", min_value=0.0, max_value=100.0, value=30.0)
        interest_rate            = st.number_input("Suku Bunga (%)", min_value=0, max_value=100, value=15)
        num_of_loan              = st.number_input("Jumlah Pinjaman", min_value=0, max_value=50, value=2)
    with c2:
        changed_credit_limit  = st.number_input("Perubahan Limit Kartu Kredit (%)", min_value=0.0, value=10.0, step=0.1)
        num_credit_inquiries  = st.number_input("Jumlah Permintaan Kredit", min_value=0, value=5)
        type_of_loan          = st.selectbox("Jenis Pinjaman", ["Pinjaman Pribadi","Pinjaman Mahasiswa","KPR",
                                                                "Pinjaman Gaji","Pinjaman Mobil","Pinjaman Pembangun Kredit",
                                                                "Konsolidasi Utang","Pinjaman Ekuitas Rumah","Tidak Ditentukan"])
        credit_mix            = st.selectbox("Komposisi Kredit", ["Buruk","Standar","Baik"])

    st.subheader("📅 Perilaku Pembayaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        delay_from_due_date    = st.number_input("Rata-rata Keterlambatan (hari)", min_value=0, value=10)
    with c2:
        num_of_delayed_payment = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, value=5)
    with c3:
        payment_of_min_amount  = st.selectbox("Hanya Bayar Minimum?", ["Tidak","Ya"])

    payment_behaviour = st.selectbox("Pola Belanja & Pembayaran", [
        "Pengeluaran Rendah - Nilai Kecil",
        "Pengeluaran Rendah - Nilai Sedang",
        "Pengeluaran Rendah - Nilai Besar",
        "Pengeluaran Tinggi - Nilai Kecil",
        "Pengeluaran Tinggi - Nilai Sedang",
        "Pengeluaran Tinggi - Nilai Besar"
    ])

    st.divider()
    submitted = st.form_submit_button("🔍 Prediksi", type="primary", use_container_width=True)

# Hasil prediksi
if submitted:
    input_data = pd.DataFrame([{
        "Age": age, "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date, "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit, "Num_Credit_Inquiries": num_credit_inquiries,
        "Outstanding_Debt": outstanding_debt, "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Total_EMI_per_month": total_emi_per_month, "Amount_invested_monthly": amount_invested_monthly,
        "Monthly_Balance": monthly_balance, "Credit_History_Age_Months": credit_history_age_months,
        "Month": month, "Occupation": occupation, "Type_of_Loan": type_of_loan,
        "Credit_Mix": credit_mix, "Payment_of_Min_Amount": payment_of_min_amount,
        "Payment_Behaviour": payment_behaviour
    }])

    model = load_model()

    try:
        pred   = model.predict(input_data)[0]
        labels = {0: "Poor", 1: "Standard", 2: "Good"}
        result = labels[pred]

        st.divider()
        if result == "Good":
            st.success(f"### ✅ Credit Score: **{result}**")
            st.caption("Risiko rendah: Layak mendapat produk kredit premium")
            st.balloons()
        elif result == "Standard":
            st.info(f"### 📊 Credit Score: **{result}**")
            st.caption("Risiko sedang: Memenuhi persyaratan dasar")
        else:
            st.error(f"### ⚠️ Credit Score: **{result}**")
            st.caption("Risiko tinggi: Perbaikan finansial disarankan")

        # Confidence bars
        try:
            clf = model.named_steps.get("classifier") or model.named_steps.get("model")
            if clf and hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(input_data)[0]
                st.write("**Tingkat Keyakinan Prediksi**")
                for label, prob in zip(["Poor", "Standard", "Good"], proba):
                    st.write(f"{label}")
                    st.progress(float(prob), text=f"{prob:.1%}")
        except Exception:
            pass

        # Ringkasan input
        with st.expander("📋 Ringkasan Input", expanded=False):
            st.dataframe(pd.DataFrame({
                "Fitur": ["Pendapatan Tahunan","Saldo Bulanan","Sisa Utang",
                          "Rasio Penggunaan Kredit","Pembayaran Terlambat","Usia Riwayat Kredit"],
                "Nilai": [f"${annual_income:,.0f}", f"${monthly_balance:,.0f}", f"${outstanding_debt:,.0f}",
                          f"{credit_utilization_ratio:.1f}%", str(num_of_delayed_payment), f"{credit_history_age_months} bulan"]
            }), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediksi gagal: {e}")
        st.caption("Pastikan model tersedia di `artifacts/RF-5_pipeline.pkl` dan semua input valid.")
