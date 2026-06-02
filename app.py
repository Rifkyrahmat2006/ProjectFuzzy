from __future__ import annotations

import streamlit as st

from fuzzy_mamdani import build_visualization_figure, evaluate_sleep_quality, load_dataset


@st.cache_data
def get_first_10_rows():
    return load_dataset().head(10)


def main() -> None:
    st.set_page_config(page_title="SLEEPZY", page_icon="💤", layout="wide")

    st.title("SLEEPZY")
    st.caption("Grafik fuzzy Mamdani untuk 10 baris pertama dari CSV")

    df = get_first_10_rows()

    if df.empty:
        st.warning("Data CSV kosong, grafik tidak dapat ditampilkan.")
        return

    st.subheader("Grafik Hasil Rule Fuzzy")
    for row_start in range(0, len(df), 2):
        left_column, right_column = st.columns(2)
        for offset, column in enumerate((left_column, right_column)):
            row_index = row_start + offset
            if row_index >= len(df):
                continue

            sample = df.iloc[row_index]
            hasil, simulation = evaluate_sleep_quality(
                float(sample["Sleep_Duration"]),
                float(sample["Physical_Activity"]),
                float(sample["Caffeine_Intake"]),
            )

            with column:
                st.markdown(
                    f"**Data {row_index + 1}**  \nDurasi: {sample['Sleep_Duration']} jam | Aktivitas: {sample['Physical_Activity']} | "
                    f"Kafein: {sample['Caffeine_Intake']} | Skor: {hasil:.2f} / 10"
                )

                figure = build_visualization_figure(
                    float(sample["Sleep_Duration"]),
                    float(sample["Physical_Activity"]),
                    float(sample["Caffeine_Intake"]),
                    simulation,
                )
                st.pyplot(figure, clear_figure=True, use_container_width=True)


if __name__ == "__main__":
    main()