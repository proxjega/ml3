import os

import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score


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


# ==============================================================================
# 2. OPTIMALAUS KLASTERIŲ SKAIČIAUS NUSTATYMAS (EMPYRINIS, ELBOW, SILUETAS)
# ==============================================================================
def elbow_point(ks, values):
    """
    Elbow taško paieška: kreivė ir tiesė tarp pirmo ir paskutinio taško normuojamos į [0, 1],
    alkūne laikomas k, kurio taškas labiausiai nutolęs nuo tos tiesės.
    """
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (values - values.min()) / (values.max() - values.min())
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    d = np.abs(dx * (y - y[0]) - dy * (x - x[0])) / np.hypot(dx, dy)
    return int(ks[np.argmax(d)])


def find_optimal_k(df, metric='euclidean', k_max=None, random_state=0,
                   out_dir='exports/k_parinkimas'):
    """
    Optimalaus klasterių skaičiaus paieška k-medoids algoritmu originalioje erdvėje:
      - empyrinis metodas: k ≈ sqrt(n/2) (naudojamas kaip viršutinė k riba),
      - Elbow metodas: k-medoids kaina (atstumų iki medoidų suma),
      - vidutinio silueto metodas.
    Prieš klasterizavimą taikomas Robust Scaling.
    """
    X = robust_scale(df, list(df.columns)).values
    n = len(X)

    k_emp = int(round(np.sqrt(n / 2)))
    print(f"Empyrinis metodas: k ≈ sqrt({n}/2) = {k_emp}")
    if k_max is None:
        k_max = k_emp

    D = pairwise_distances(X, metric=metric)

    rows = []
    for k in range(2, k_max + 1):
        res = kmedoids.fasterpam(D, k, random_state=random_state)
        labels = res.labels
        rows.append({
            'k': k,
            'cost': res.loss,
            'silhouette': silhouette_score(D, labels, metric='precomputed'),
        })
    tab = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/kmedoids_{metric}_k_lentele.csv", index=False)

    def plot_k(sub, suffix):
        best = {
            'cost': elbow_point(sub['k'].values, sub['cost'].values),
            'silhouette': int(sub.loc[sub['silhouette'].idxmax(), 'k']),
        }
        titles = {
            'cost': f"Elbow (alkūnė ties k={best['cost']})",
            'silhouette': f"Vidutinis siluetas ↑ (geriausias k={best['silhouette']})",
        }
        ylabels = {'cost': 'Atstumų iki medoidų suma', 'silhouette': 'Vidutinis siluetas'}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, col in zip(axes, ['cost', 'silhouette']):
            ax.plot(sub['k'], sub[col], 'o-', markersize=3)
            ax.axvline(best[col], color='red', linestyle='--', linewidth=1)
            ax.set_title(titles[col])
            ax.set_xlabel('Klasterių skaičius k')
            ax.set_ylabel(ylabels[col])
            ax.grid(alpha=0.3)
        fig.suptitle(f"Optimalaus klasterių skaičiaus nustatymas (k-medoids, {metric} atstumas, "
                     f"k = 2…{sub['k'].max()})")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/kmedoids_{metric}_k_parinkimas{suffix}.png", dpi=150)
        plt.close(fig)
        return best

    # Visas intervalas k = 2…k_max ir priartintas vaizdas k = 2…15
    best = plot_k(tab, '')
    best_15 = plot_k(tab[tab['k'] <= 15], '_iki15')

    print(tab.round(4).to_string(index=False))
    print(f"Geriausi k (k = 2…{k_max}):", best)
    print("Geriausi k (k = 2…15):", best_15)
    return tab, best, best_15


# ==============================================================================
# 3. K-MEDOIDS KLASTERIŲ VIZUALIZACIJA PCA ERDVĖJE
# ==============================================================================
def plot_kmedoids_pca(df, k=4, metric='euclidean', random_state=0,
                      out_dir='exports/vizualizacija'):
    """
    Suklasterizuoja duomenis k-medoids algoritmu originalioje (Robust Scaling) erdvėje
    ir pavaizduoja klasterius pirmųjų dviejų PCA komponenčių plokštumoje.
    Kairėje – visi objektai, dešinėje – priartintas vaizdas (1–99 procentilių ribos).
    """
    X = robust_scale(df, list(df.columns)).values
    D = pairwise_distances(X, metric=metric)
    res = kmedoids.fasterpam(D, k, random_state=random_state)
    labels = res.labels

    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_ * 100

    sizes = np.bincount(labels, minlength=k)
    print(f"k-medoids (k={k}, {metric}) klasterių dydžiai:", dict(enumerate(sizes)))
    print(f"PCA paaiškinta dispersija: PC1={evr[0]:.1f} %, PC2={evr[1]:.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, zoom in zip(axes, [False, True]):
        for c in range(k):
            m = labels == c
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, label=f"Klasteris {c} (n={sizes[c]})")
        ax.scatter(Z[res.medoids, 0], Z[res.medoids, 1], marker='X', s=120,
                   c='black', edgecolors='white', label='Medoidai')
        if zoom:
            lo, hi = np.percentile(Z, [1, 99], axis=0)
            pad = (hi - lo) * 0.05
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_title("Priartintas vaizdas (1–99 procentilės)")
        else:
            ax.set_title("Visi objektai")
        ax.set_xlabel(f"PC1 ({evr[0]:.1f} %)")
        ax.set_ylabel(f"PC2 ({evr[1]:.1f} %)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, markerscale=2)
    fig.suptitle(f"k-medoids klasteriai PCA erdvėje (k={k}, {metric} atstumas)")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/kmedoids_{metric}_k{k}_pca.png", dpi=150)
    plt.close(fig)
    return labels


if __name__ == '__main__':
    df_out = prepare_dataset()
    print(f"Išsaugota exports/duomenys/A28_cleaned.csv: {df_out.shape[0]} eilučių, {df_out.shape[1]} požymių")

    find_optimal_k(df_out, metric='euclidean')

    plot_kmedoids_pca(df_out, k=4, metric='euclidean')
