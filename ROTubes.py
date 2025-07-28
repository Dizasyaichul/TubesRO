import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from streamlit_option_menu import option_menu

class ComprehensiveDietOptimizer:
    def __init__(self):
        self.M = 1000000
        self.setup_dataset()

    def setup_dataset(self):
        self.foods = pd.DataFrame({
            'food_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'food_name': ['Nasi Putih', 'Ayam Goreng', 'Sayur Bayam', 'Telur Rebus',
                          'Susu Sapi', 'Roti Gandum', 'Ikan Salmon', 'Pisang'],
            'cost_per_unit': [5000, 15000, 3000, 2500, 8000, 4000, 25000, 2000],
            'unit': ['porsi', 'potong', 'porsi', 'butir', 'gelas', 'lembar', 'porsi', 'buah']
        })

        self.nutrition = pd.DataFrame({
            'food_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'protein_g': [2.7, 25.0, 2.9, 6.3, 3.2, 3.6, 22.0, 1.1],
            'carbs_g': [28.0, 0.0, 3.6, 0.6, 4.8, 15.0, 0.0, 27.0],
            'fat_g': [0.3, 14.0, 0.4, 5.3, 3.3, 1.0, 13.0, 0.3],
            'fiber_g': [0.4, 0.0, 2.2, 0.0, 0.0, 2.0, 0.0, 3.1],
            'calcium_mg': [10, 15, 99, 25, 113, 30, 12, 5],
            'iron_mg': [0.8, 1.3, 2.7, 0.6, 0.1, 1.5, 0.8, 0.3],
            'vitamin_a_iu': [0, 100, 4690, 160, 126, 0, 59, 64],
            'vitamin_c_mg': [0, 0, 28, 0, 1, 0, 0, 8.7],
            'calories': [130, 250, 23, 78, 61, 80, 208, 105]
        })

        self.requirements = pd.DataFrame({
            'nutrient': ['protein', 'carbs', 'fat', 'fiber', 'calcium',
                         'iron', 'vitamin_a', 'vitamin_c', 'calories'],
            'min_requirement': [50, 225, 44, 25, 1000, 8, 700, 65, 1800],
            'max_requirement': [150, 325, 78, 35, 2500, 45, 3000, 2000, 2500],
            'unit': ['g', 'g', 'g', 'g', 'mg', 'mg', 'IU', 'mg', 'kcal']
        })

    def solve_with_big_m(self, excluded_foods=None):
        if excluded_foods is None:
            excluded_foods = []

        allowed_foods = self.foods[~self.foods['food_name'].isin(excluded_foods)].reset_index(drop=True)
        allowed_ids = allowed_foods['food_id']
        allowed_nutrition = self.nutrition[self.nutrition['food_id'].isin(allowed_ids)].reset_index(drop=True)

        costs = list(allowed_foods['cost_per_unit'])
        nutrient_cols = ['protein_g', 'carbs_g', 'fat_g', 'fiber_g', 'calcium_mg',
                         'iron_mg', 'vitamin_a_iu', 'vitamin_c_mg', 'calories']

        n_foods = len(allowed_foods)
        n_nutrients = len(self.requirements)

        c = [cost for cost in costs] + [0]*n_nutrients + [self.M]*n_nutrients

        A_eq, b_eq = [], []
        for i in range(n_nutrients):
            coeffs = list(allowed_nutrition[nutrient_cols[i]].values)
            slack = [0]*n_nutrients; slack[i] = -1
            artificial = [0]*n_nutrients; artificial[i] = 1
            A_eq.append(coeffs + slack + artificial)
            b_eq.append(self.requirements.iloc[i]['min_requirement'])

        A_ub, b_ub = [], []
        for i in range(n_nutrients):
            coeffs = list(allowed_nutrition[nutrient_cols[i]].values)
            A_ub.append(coeffs + [0]*(2*n_nutrients))
            b_ub.append(self.requirements.iloc[i]['max_requirement'])

        bounds = [(0, None)] * (n_foods + 2*n_nutrients)
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        steps = []
        final_x = result.x[:n_foods] if result.success else np.zeros(n_foods)
        for i in range(6):
            fraction = i / 5
            x_iter = final_x * fraction
            step_detail = {
                'Iterasi': f'Iterasi {i}',
                'Total Biaya': np.dot(x_iter, allowed_foods['cost_per_unit'])
            }
            for j in range(n_foods):
                step_detail[allowed_foods['food_name'][j]] = round(x_iter[j], 2)
            steps.append(step_detail)

        steps_df = pd.DataFrame(steps)
        steps_df['Total Biaya'] = steps_df['Total Biaya'].apply(lambda x: f"Rp {x:,.2f}")
        for col in allowed_foods['food_name']:
            if col in steps_df.columns:
                steps_df[col] = steps_df[col].map(lambda x: f"{x:.2f}")

        self.iterasi_df = steps_df
        self.filtered_foods = allowed_foods
        return result, nutrient_cols, n_foods, n_nutrients, self.iterasi_df, self.filtered_foods

    def display_solution(self, result, foods, n_foods):
        if result.success:
            x = result.x[:n_foods]
            diet = []
            nutrition_df = self.nutrition[self.nutrition['food_id'].isin(foods['food_id'])].reset_index(drop=True)

            for i in range(n_foods):
                if x[i] > 0.01:
                    food = foods.iloc[i]
                    nutrisi = nutrition_df.iloc[i]
                    jumlah = x[i]

                    # Hitung total nutrisi skala normalisasi
                    total_nutrisi = (
                        nutrisi['protein_g'] +
                        nutrisi['carbs_g'] +
                        nutrisi['fat_g'] +
                        nutrisi['fiber_g'] +
                        nutrisi['calcium_mg'] / 1000 +
                        nutrisi['iron_mg'] / 1000 +
                        nutrisi['vitamin_a_iu'] / 10000 +
                        nutrisi['vitamin_c_mg'] / 100 +
                        nutrisi['calories'] / 1000
                    ) * jumlah

                    diet.append({
                        'Makanan': food['food_name'],
                        'Jumlah': round(jumlah, 2),
                        'Unit': food['unit'],
                        'Biaya': round(jumlah * food['cost_per_unit'], 2),
                        'Total Nutrisi': round(total_nutrisi, 2)
                    })
            return pd.DataFrame(diet)
        else:
            return None

st.set_page_config(page_title="Optimasi Diet - Big M", layout="wide")
st.title("🍱 Optimasi Makananan Diet Bergizi dengan Metode Big-M")

with st.sidebar:
    selected_tab = option_menu(
        menu_title="Menu Utama",
        options=["Hasil Optimasi", "Iterasi Solusi"],
        icons=["graph-up", "arrow-repeat"],
        menu_icon="list",
        default_index=0,
        orientation="vertical"
    )

    optimizer = ComprehensiveDietOptimizer()
    excluded_foods = st.multiselect(
        "Pilih makanan yang tidak ingin dikonsumsi:",
        options=optimizer.foods['food_name'].tolist()
    )

    if len(excluded_foods) == len(optimizer.foods):
        st.error("❌ Anda telah mengecualikan semua makanan.")
        st.stop()

result, nutrient_cols, n_foods, n_nutrients, iterasi_df, filtered_foods = optimizer.solve_with_big_m(
    excluded_foods=excluded_foods
)
diet_df = optimizer.display_solution(result, filtered_foods, n_foods)
total_cost = np.dot(result.x[:n_foods], filtered_foods['cost_per_unit'].values)

if result.success:
    if selected_tab == "Hasil Optimasi":
        st.success("✅ Solusi ditemukan!")
        st.subheader("💡 Rekomendasi Komposisi Diet Optimal")
        st.dataframe(diet_df, hide_index=True)

        st.subheader("💸 Total Biaya")
        st.metric("Biaya Minimum", f"Rp {total_cost:,.2f}")

    elif selected_tab == "Iterasi Solusi":
        st.subheader("🔁 Simulasi Iterasi Menuju Solusi Optimal")
        st.dataframe(iterasi_df, hide_index=True)
else:
    st.error("❌ Optimasi gagal. Solusi tidak ditemukan.")






