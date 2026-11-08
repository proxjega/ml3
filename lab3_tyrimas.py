import os
import sys

import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
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
DISTANCES = {
    'Minkowski p=0.5': ('minkowski', {'p': 0.5}),
    'Manhattan (p=1)': ('cityblock', {}),
    'Euclidean (p=2)': ('euclidean', {}),
    'Minkowski p=3': ('minkowski', {'p': 3}),
    'Chebyshev (p=inf)': ('chebyshev', {}),
    'Cosine': ('cosine', {}),
}


def compare_distances(df, ks=(2, 3, 4, 6), random_state=0, out_dir='exports/k_parinkimas'):
    """
    Atstumo mato įtakos išskirtims tyrimas k-medoids algoritmu (Robust Scaling erdvėje).
    Lentelė: eilutėse – atstumai, stulpeliuose – k, langeliuose – klasterių dydžiai
    (mažėjimo tvarka). Mažesnis Minkovskio p mažina vieno požymio ekstremalių skirtumų įtaką;
    p < 1 nėra tikras atstumas (negalioja trikampio nelygybė).
    """
    X = robust_scale(df, list(df.columns)).values

    tab = pd.DataFrame(index=pd.Index(list(DISTANCES), name='Atstumas'),
                       columns=[f'k = {k}' for k in ks])
    for name, (metric, kw) in DISTANCES.items():
        D = squareform(pdist(X, metric=metric, **kw))
        for k in ks:
            labels = kmedoids.fasterpam(D, k, random_state=random_state).labels
            sizes = np.sort(np.bincount(labels, minlength=k))[::-1]
            tab.loc[name, f'k = {k}'] = ', '.join(map(str, sizes))

    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/kmedoids_atstumu_palyginimas.csv")
    print("k-medoids klasterių dydžiai su skirtingais atstumais:")
    print(tab.to_string())
    return tab


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


def find_optimal_k(df, metrics=('Cosine', 'Minkowski p=0.5'), k_max=None, random_state=0,
                   out_dir='exports/k_parinkimas'):
    """
    Optimalaus klasterių skaičiaus paieška k-medoids algoritmu originalioje erdvėje:
      - empirinis metodas: k ≈ sqrt(n/2) (naudojamas kaip viršutinė k riba),
      - Elbow metodas: k-medoids kaina (atstumų iki medoidų suma),
      - vidutinio silueto metodas.
    Skaičiuojama kiekvienam atstumui iš `metrics` (raktai iš DISTANCES).
    Prieš klasterizavimą taikomas Robust Scaling.
    Sukuriami du paveikslai: vidutinis siluetas (k = 2…k_max) ir Elbow (eilutės – atstumai,
    stulpeliai – k = 2…k_max ir k = 2…15).
    """
    X = robust_scale(df, list(df.columns)).values
    n = len(X)

    k_emp = int(round(np.sqrt(n / 2)))
    print(f"Empyrinis metodas: k ≈ sqrt({n}/2) = {k_emp}")
    if k_max is None:
        k_max = k_emp

    rows = []
    for name in metrics:
        metric, kw = DISTANCES[name]
        D = squareform(pdist(X, metric=metric, **kw))
        for k in range(2, k_max + 1):
            res = kmedoids.fasterpam(D, k, random_state=random_state)
            rows.append({
                'atstumas': name,
                'k': k,
                'cost': res.loss,
                'silhouette': silhouette_score(D, res.labels, metric='precomputed'),
            })
    tab = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/kmedoids_k_lentele.csv", index=False)

    def sub_of(name, k_to):
        return tab[(tab['atstumas'] == name) & (tab['k'] <= k_to)]

    best = {name: {} for name in metrics}

    # 1 paveikslas: vidutinis siluetas, k = 2…k_max (po vieną grafiką kiekvienam atstumui)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    for ax, name in zip(axes[0], metrics):
        sub = sub_of(name, k_max)
        k_best = int(sub.loc[sub['silhouette'].idxmax(), 'k'])
        best[name]['silhouette'] = k_best
        ax.plot(sub['k'], sub['silhouette'], 'o-', markersize=3)
        ax.axvline(k_best, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"{name} (geriausias k={k_best})")
        ax.set_xlabel('Klasterių skaičius k')
        ax.set_ylabel('Vidutinis siluetas')
        ax.grid(alpha=0.3)
    fig.suptitle(f"Vidutinio silueto metodas (k-medoids, k = 2…{k_max})")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/kmedoids_siluetas.png", dpi=150)
    plt.close(fig)

    # 2 paveikslas: Elbow, eilutėse – atstumai, stulpeliuose – k = 2…k_max ir k = 2…15
    k_ranges = [k_max, 15]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 4.2 * len(metrics)), squeeze=False)
    for row, name in zip(axes, metrics):
        for ax, k_to in zip(row, k_ranges):
            sub = sub_of(name, k_to)
            k_best = elbow_point(sub['k'].values, sub['cost'].values)
            best[name][f'elbow (k ≤ {k_to})'] = k_best
            ax.plot(sub['k'], sub['cost'], 'o-', markersize=3)
            ax.axvline(k_best, color='red', linestyle='--', linewidth=1)
            ax.set_title(f"{name}, k = 2…{k_to} (alkūnė ties k={k_best})")
            ax.set_xlabel('Klasterių skaičius k')
            ax.set_ylabel('Atstumų iki medoidų suma')
            ax.grid(alpha=0.3)
    fig.suptitle("Elbow metodas (k-medoids)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/kmedoids_elbow.png", dpi=150)
    plt.close(fig)

    print(tab.round(4).to_string(index=False))
    print("Geriausi k:", best)
    return tab, best


def plot_dendrogram(df, method='single', distance='Cosine', k_max=15, p=30,
                    out_dir='exports/k_parinkimas'):
    """
    AGNES (aglomeracinio hierarchinio klasterizavimo) dendrograma.
    distance – atstumo pavadinimas iš DISTANCES (Ward jungimas galimas tik su Euklido atstumu).
    Prieš klasterizavimą taikomas Robust Scaling.
    Optimalus k parenkamas pagal didžiausią šuolį tarp paskutinių sujungimų aukščių:
    dendrograma kerpama per šio šuolio vidurį.
    Rodomi tik paskutiniai p sujungimai (truncate_mode='lastp'), nes objektų yra 4455.
    """
    X = robust_scale(df, list(df.columns)).values
    metric, kw = DISTANCES[distance]
    Z = linkage(pdist(X, metric=metric, **kw), method=method)
    tag = f"{method}_{metric}" + (f"_p{kw['p']}" if 'p' in kw else '')

    # Sujungimų aukščiai mažėjimo tvarka: tarp h[i-1] ir h[i] kirpimas duoda i+1 klasterių
    h = Z[::-1, 2][:k_max]
    gaps = h[:-1] - h[1:]
    best_i = int(np.argmax(gaps)) + 1
    k_best = best_i + 1
    cut = (h[best_i - 1] + h[best_i]) / 2

    print(f"Dendrograma ({method}, {distance}): didžiausias šuolis ties k={k_best}, "
          f"kirpimo aukštis={cut:.4g}")
    print("Paskutinių sujungimų aukščiai:", np.array2string(h, precision=4))

    # Single jungimas sukuria labai gilų (grandininį) medį – scipy piešimui reikia didesnės rekursijos ribos
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    dendrogram(Z, truncate_mode='lastp', p=p, show_contracted=True, ax=ax,
               color_threshold=cut, above_threshold_color='grey')
    ax.axhline(cut, color='red', linestyle='--', linewidth=1,
               label=f"Kirpimas: k={k_best} (aukštis {cut:.4g})")
    ax.set_title(f"AGNES dendrograma ({method} jungimas, {distance} atstumas, paskutiniai {p} sujungimai)")
    ax.set_xlabel("Objektai / sujungtų objektų grupės (skliaustuose – objektų skaičius)")
    ax.set_ylabel("Sujungimo atstumas")
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/agnes_{tag}_dendrograma.png", dpi=150)
    plt.close(fig)
    return Z, k_best


# ==============================================================================
# 3. KLASTERIŲ VIZUALIZACIJA PCA ERDVĖJE (K-MEDOIDS, AGNES)
# ==============================================================================
def plot_clusters_pca(X, labels, title, out_path, centers=None, random_state=0):
    """
    Pavaizduoja klasterius pirmųjų dviejų PCA komponenčių plokštumoje.
    Kairėje – visi objektai, dešinėje – priartintas vaizdas (1–99 procentilių ribos).
    centers – klasterių centrų (pvz., medoidų) indeksai, jei jie yra.
    """
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_ * 100

    clusters = np.unique(labels)
    sizes = {int(c): int((labels == c).sum()) for c in clusters}
    print(f"{title} – klasterių dydžiai:", sizes)
    print(f"PCA paaiškinta dispersija: PC1={evr[0]:.1f} %, PC2={evr[1]:.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, zoom in zip(axes, [False, True]):
        for c in clusters:
            m = labels == c
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, label=f"Klasteris {c} (n={sizes[c]})")
        if centers is not None:
            ax.scatter(Z[centers, 0], Z[centers, 1], marker='X', s=120,
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
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kmedoids_pca(df, k=4, metric='euclidean', random_state=0,
                      out_dir='exports/vizualizacija'):
    """
    Suklasterizuoja duomenis k-medoids algoritmu originalioje (Robust Scaling) erdvėje
    ir pavaizduoja klasterius PCA plokštumoje.
    """
    X = robust_scale(df, list(df.columns)).values
    D = pairwise_distances(X, metric=metric)
    res = kmedoids.fasterpam(D, k, random_state=random_state)
    labels = res.labels

    plot_clusters_pca(X, labels, f"k-medoids klasteriai PCA erdvėje (k={k}, {metric} atstumas)",
                      f"{out_dir}/kmedoids_{metric}_k{k}_pca.png",
                      centers=res.medoids, random_state=random_state)
    return labels


def plot_agnes_pca(df, k=4, method='single', out_dir='exports/vizualizacija'):
    """
    Suklasterizuoja duomenis AGNES algoritmu (Euklido atstumas) originalioje (Robust Scaling)
    erdvėje, dendrogramą kerpa į k klasterių ir pavaizduoja klasterius PCA plokštumoje.
    """
    X = robust_scale(df, list(df.columns)).values
    metric, kw = DISTANCES[distance]
    Z = linkage(pdist(X, metric=metric, **kw), method=method)
    tag = f"{method}_{metric}" + (f"_p{kw['p']}" if 'p' in kw else '')
    labels = fcluster(Z, t=k, criterion='maxclust') - 1   # klasteriai numeruojami nuo 0

    plot_clusters_pca(X, labels, f"AGNES klasteriai PCA erdvėje (k={k}, {method} jungimas, euclidean atstumas)",
                      f"{out_dir}/agnes_{method}_euclidean_k{k}_pca.png")
    return labels


if __name__ == '__main__':
    df_out = prepare_dataset()
    print(f"Išsaugota exports/duomenys/A28_cleaned.csv: {df_out.shape[0]} eilučių, {df_out.shape[1]} požymių")

    compare_distances(df_out)
    find_optimal_k(df_out, metrics=('Cosine', 'Minkowski p=0.5'))
    plot_dendrogram(df_out, method='single', distance='Cosine')

    plot_kmedoids_pca(df_out, k=4, metric='euclidean')
    plot_agnes_pca(df_out, k=2, method='single')
