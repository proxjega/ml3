import os

import pandas as pd


# ==============================================================================
# 1. DUOMENŲ PARUOŠIMAS (DUBLIKATAI, TRŪKSTAMOS REIKŠMĖS, SPEARMAN KORELIACIJA)
# ==============================================================================
def robust_scale(df_in, cols):
    df_out = df_in[cols].copy()
    for c in cols:
        med = df_out[c].median()
        iqr = df_out[c].quantile(0.75) - df_out[c].quantile(0.25)
        df_out[c] = (df_out[c] - med) / iqr if iqr != 0 else 0.0
    return df_out


def prepare_dataset(csv_path='A28.csv', out_path='exports/duomenys/A28_cleaned.csv'):
    """
    Paruošia išvalytą duomenų aibę:
      - pašalinami dublikatai,
      - trūkstamos reikšmės užpildomos mediana pagal klasę,
      - pašalinami mažo kintamumo / mažo Eta2 požymiai (CV < 0.01 ir Eta2 < 0.01),
      - pašalinami stipriai koreliuoti požymiai (Spearman |rho| > 0.95),
      - pašalinamas 'class' stulpelis.
    Robust scaling bus taikomas veliau (kadangi stabilumo eksperimentui bus imama 80% imties).
    """
    df = pd.read_csv(csv_path)
    df['V01'] = df['V01'].astype(str).str.replace('value=', '', regex=False).astype(float)
    feature_cols = [c for c in df.columns if c != 'class']

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Pašalinami dublikatai ir trūkstamos reikšmės užpildomos mediana pagal klasę
    df_clean = df.drop_duplicates().reset_index(drop=True)
    for col in feature_cols:
        df_clean[col] = df_clean.groupby('class')[col].transform(lambda x: x.fillna(x.median()))

    # 1 žingsnis: Mažo kintamumo ir mažo koreliacijos santykio šalinimas (CV < 0.01 ir Eta2 < 0.01)
    dropped_step1 = []
    eta2_dict = {}
    for col in feature_cols:
        series = df_clean[col]
        mean_val, std_val = series.mean(), series.std()
        cv = std_val / abs(mean_val) if mean_val != 0 else 0.0
        ss_total = ((series - mean_val) ** 2).sum()
        class_stats = df_clean.groupby('class')[col].agg(['count', 'mean'])
        ss_between = (class_stats['count'] * ((class_stats['mean'] - mean_val) ** 2)).sum()
        eta2 = (ss_between / ss_total) if ss_total != 0 else 0.0
        eta2_dict[col] = eta2
        if cv < 0.01 and eta2 < 0.01:
            dropped_step1.append(col)

    cols_B = [c for c in feature_cols if c not in dropped_step1]

    # 2 žingsnis: Stipriai koreliuojančių požymių filtras naudojant Spearman koreliaciją (|rho| > 0.95)
    corr_matrix = df_clean[cols_B].corr(method='spearman').abs()
    dropped_step2 = set()
    for i in range(len(cols_B)):
        for j in range(i + 1, len(cols_B)):
            c1, c2 = cols_B[i], cols_B[j]
            if c1 in dropped_step2 or c2 in dropped_step2:
                continue
            if corr_matrix.loc[c1, c2] > 0.95:
                if eta2_dict[c1] >= eta2_dict[c2]:
                    dropped_step2.add(c2)
                else:
                    dropped_step2.add(c1)

    cols_A = [c for c in cols_B if c not in dropped_step2]

    # Išsaugoma išvalyta aibė be 'class' stulpelio ir be robust scaling
    df_out = df_clean[cols_A]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return df_out


if __name__ == '__main__':
    df_out = prepare_dataset()
    print(f"Išsaugota exports/duomenys/A28_cleaned.csv: {df_out.shape[0]} eilučių, {df_out.shape[1]} požymių")
