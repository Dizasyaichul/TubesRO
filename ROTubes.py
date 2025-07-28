import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from streamlit_option_menu import option_menu

class OptimasiDietMahasiswa:
    def __init__(self):
        self.M = 1000000
        self.atur_dataset()

    def atur_dataset(self):
        self.makanan = pd.DataFrame({
            'id_makanan': [1, 2, 3, 4, 5, 6, 7, 8],
            'nama_makanan': ['Nasi Putih', 'Ayam Goreng', 'Sayur Bayam', 'Telur Rebus',
                             'Susu Sapi', 'Roti Gandum', 'Ikan Salmon', 'Pisang'],
            'biaya_per_unit': [5000, 15000, 3000, 2500, 8000, 4000, 25000, 2000],
            'satuan': ['porsi', 'potong', 'porsi', 'butir', 'gelas', 'lembar', 'porsi', 'buah']
        })

        self.kandungan_gizi = pd.DataFrame({
            'id_makanan': [1, 2, 3, 4, 5, 6, 7, 8],
            'protein': [2.7, 25.0, 2.9, 6.3, 3.2, 3.6, 22.0, 1.1],
            'karbo': [28.0, 0.0, 3.6, 0.6, 4.8, 15.0, 0.0, 27.0],
            'lemak': [0.3, 14.0, 0.4, 5.3, 3.3, 1.0, 13.0, 0.3],
            'serat': [0.4, 0.0, 2.2, 0.0, 0.0, 2.0, 0.0, 3.1],
            'kalsium': [10, 15, 99, 25, 113, 30, 12, 5],
            'zat_besi': [0.8, 1.3, 2.7, 0.6, 0.1, 1.5, 0.8, 0.3],
            'vitamin_a': [0, 100, 4690, 160, 126, 0, 59, 64],
            'vitamin_c': [0, 0, 28, 0, 1, 0, 0, 8.7],
            'kalori': [130, 250, 23, 78, 61, 80, 208, 105]
        })

        self.kebutuhan_gizi = pd.DataFrame({
            'gizi': ['protein', 'karbo', 'lemak', 'serat', 'kalsium',
                     'zat_besi', 'vitamin_a', 'vitamin_c', 'kalori'],
            'minimal': [50, 225, 44, 25, 1000, 8, 700, 65, 1800],
            'maksimal': [150, 325, 78, 35, 2500, 45, 3000, 2000, 2500],
            'satuan': ['g', 'g', 'g', 'g', 'mg', 'mg', 'IU', 'mg', 'kcal']
        })

    def jalankan_big_m(self, makanan_dikecualikan=None):
        if makanan_dikecualikan is None:
            makanan_dikecualikan = []

        makanan_boleh = self.makanan[~self.makanan['nama_makanan'].isin(makanan_dikecualikan)].reset_index(drop=True)
        id_terpilih = makanan_boleh['id_makanan']
        gizi_terpilih = self.kandungan_gizi[self.kandungan_gizi['id_makanan'].isin(id_terpilih)].reset_index(drop=True)

        biaya = list(makanan_boleh['biaya_per_unit'])
        kolom_gizi = ['protein', 'karbo', 'lemak', 'serat', 'kalsium',
                      'zat_besi', 'vitamin_a', 'vitamin_c', 'kalori']

        jumlah_makanan = len(makanan_boleh)
        jumlah_gizi = len(self.kebutuhan_gizi)

        c = biaya + [0]*jumlah_gizi + [self.M]*jumlah_gizi

        A_eq, b_eq = [], []
        for i in range(jumlah_gizi):
            koef = list(gizi_terpilih[kolom_gizi[i]].values)
            slack = [0]*jumlah_gizi; slack[i] = -1
            artificial = [0]*jumlah_gizi; artificial[i] = 1
            A_eq.append(koef + slack + artificial)
            b_eq.append(self.kebutuhan_gizi.iloc[i]['minimal'])

        A_ub, b_ub = [], []
        for i in range(jumlah_gizi):
            koef = list(gizi_terpilih[kolom_gizi[i]].values)
            A_ub.append(koef + [0]*(2*jumlah_gizi))
            b_ub.append(self.kebutuhan_gizi.iloc[i]['maksimal'])

        batas = [(0, None)] * (jumlah_makanan + 2*jumlah_gizi)
        hasil = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=batas, method='highs')

        langkah = []
        solusi_akhir = hasil.x[:jumlah_makanan] if hasil.success else np.zeros(jumlah_makanan)
        for i in range(6):
            fraksi = i / 5
            iter_x = solusi_akhir * fraksi
            detail = {
                'Iterasi': f'Iterasi {i}',
                'Total Biaya': np.dot(iter_x, makanan_boleh['biaya_per_unit'])
            }
            for j in range(jumlah_makanan):
                detail[makanan_boleh['nama_makanan'][j]] = round(iter_x[j], 2)
            langkah.append(detail)

        df_langkah = pd.DataFrame(langkah)
        df_langkah['Total Biaya'] = df_langkah['Total Biaya'].apply(lambda x: f"Rp {x:,.2f}")
        for kolom in makanan_boleh['nama_makanan']:
            if kolom in df_langkah.columns:
                df_langkah[kolom] = df_langkah[kolom].map(lambda x: f"{x:.2f}")

        self.data_iterasi = df_langkah
        self.makanan_terfilter = makanan_boleh
        return hasil, kolom_gizi, jumlah_makanan, jumlah_gizi, self.data_iterasi, self.makanan_terfilter

    def tampilkan_solusi(self, hasil, makanan, jumlah_makanan):
        if hasil.success:
            x = hasil.x[:jumlah_makanan]
            rekomendasi = []
            gizi_df = self.kandungan_gizi[self.kandungan_gizi['id_makanan'].isin(makanan['id_makanan'])].reset_index(drop=True)

            for i in range(jumlah_makanan):
                if x[i] > 0.01:
                    item = makanan.iloc[i]
                    gizi = gizi_df.iloc[i]
                    jumlah = x[i]

                    total = (
                        gizi['protein'] +
                        gizi['karbo'] +
                        gizi['lemak'] +
                        gizi['serat'] +
                        gizi['kalsium'] / 1000 +
                        gizi['zat_besi'] / 1000 +
                        gizi['vitamin_a'] / 10000 +
                        gizi['vitamin_c'] / 100 +
                        gizi['kalori'] / 1000
                    ) * jumlah

                    rekomendasi.append({
                        'Makanan': item['nama_makanan'],
                        'Jumlah': round(jumlah, 2),
                        'Unit': item['satuan'],
                        'Biaya': round(jumlah * item['biaya_per_unit'], 2),
                        'Total Nutrisi': round(total, 2)
                    })
            return pd.DataFrame(rekomendasi)
        else:
            return None

# UI Streamlit
st.set_page_config(page_title="Optimasi Diet - Big M", layout="wide")
st.title("🍱 Optimasi Makanan Diet Bergizi dengan Metode Big-M")

with st.sidebar:
    menu_pilihan = option_menu(
        menu_title="Menu Utama",
        options=["Hasil Optimasi", "Iterasi Solusi"],
        icons=["graph-up", "arrow-repeat"],
        menu_icon="list",
        default_index=0,
        orientation="vertical"
    )

    model = OptimasiDietMahasiswa()
    makanan_dikecualikan = st.multiselect(
        "Pilih makanan yang ingin dikecualikan:",
        options=model.makanan['nama_makanan'].tolist()
    )

    if len(makanan_dikecualikan) == len(model.makanan):
        st.error("❌ Semua makanan tidak boleh dikecualikan.")
        st.stop()

hasil, kolom_gizi, jml_makanan, jml_gizi, iterasi_df, makanan_terfilter = model.jalankan_big_m(
    makanan_dikecualikan=makanan_dikecualikan
)
df_solusi = model.tampilkan_solusi(hasil, makanan_terfilter, jml_makanan)
total_biaya = np.dot(hasil.x[:jml_makanan], makanan_terfilter['biaya_per_unit'].values)

if hasil.success:
    if menu_pilihan == "Hasil Optimasi":
        st.success("✅ Solusi ditemukan!")
        st.subheader("💡 Rekomendasi Diet Optimal")
        st.dataframe(df_solusi, hide_index=True)

        st.subheader("💸 Total Biaya")
        st.metric("Biaya Minimum", f"Rp {total_biaya:,.2f}")

    elif menu_pilihan == "Iterasi Solusi":
        st.subheader("🔁 Proses Iterasi Menuju Solusi")
        st.dataframe(iterasi_df, hide_index=True)
else:
    st.error("❌ Gagal menemukan solusi.")










