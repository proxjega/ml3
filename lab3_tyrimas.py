import hashlib
import os
import sys
import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import pair_confusion_matrix


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


def prepare_dataset(csv_path='A28.csv', out_path='exports/duomenys/A28_cleaned.csv', outlier_k=3.0):
    """
    Paruošia išvalytą duomenų aibę:
      - pašalinami dublikatai,
      - trūkstamos reikšmės užpildomos mediana pagal klasę,
      - ekstremalios išskirtys (už Q1 - k*IQR, Q3 + k*IQR ribų, k=3) pakeičiamos ribine reikšme,
      - pašalinami mažo kintamumo / mažo Eta2 požymiai (CV < 0.01 ir Eta2 < 0.01),
      - pašalinami stipriai koreliuoti požymiai (Spearman |rho| > 0.95),
      - 'class' stulpelis atskiriamas nuo požymių (klasterizavime nenaudojamas, bet
        išsaugomas klasterių palyginimui su tikrosiomis klasėmis).
    Grąžina (požymių DataFrame, klasių Series, išskirčių kaukė).
    Išskirčių kaukė – bool Series: True, jei objekto bent viena galutinai palikto požymio
    reikšmė buvo ekstremali išskirtis ir pakeista ribine reikšme. Ji naudojama vizualizacijose
    išskirtims pažymėti ir išsaugoma kaip stulpelis 'isskirtis'.
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

    # Ekstremalios išskirtys pakeičiamos artimiausia ribine reikšme (winsorizavimas)
    q1 = df_clean[feature_cols].quantile(0.25)
    q3 = df_clean[feature_cols].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - outlier_k * iqr, q3 + outlier_k * iqr
    outlier_cells = (df_clean[feature_cols] < lower) | (df_clean[feature_cols] > upper)
    n_outliers = outlier_cells.sum()
    df_clean[feature_cols] = df_clean[feature_cols].clip(lower=lower, upper=upper, axis=1)
    print(f"Ekstremalios išskirtys (k={outlier_k}): pakeista {int(n_outliers.sum())} reikšmių")

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

    # Išsaugoma išvalyta aibė be robust scaling; 'class' paliekamas paskutiniame stulpelyje,
    # bet į klasterizavimui naudojamus požymius neįtraukiamas
    df_out = df_clean[cols_A]
    y = df_clean['class']
    # Objektai, kurių bent viena paliktų požymių reikšmė buvo pakeista (išskirtis)
    outliers = outlier_cells[cols_A].any(axis=1).rename('isskirtis')
    print(f"Objektų su išskirtimis (palikti požymiai): {int(outliers.sum())} iš {len(outliers)}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.concat([df_clean[cols_A + ['class']], outliers], axis=1).to_csv(out_path, index=False)
    return df_out, y, outliers


# ==============================================================================
# 2. OPTIMALAUS KLASTERIŲ SKAIČIAUS NUSTATYMAS (EMPYRINIS, ELBOW, SILUETAS)
# ==============================================================================
# Visur naudojamas Euklido atstumas (Robust Scaling erdvėje)
def euclidean_matrix(X):
    """Euklido atstumų matrica n×n."""
    return squareform(pdist(X, metric='euclidean'))


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


def find_optimal_k(df, k_max=None, random_state=0, min_samples=None, dbscan_eps=None,
                   out_dir='exports/k_parinkimas'):
    """
    Optimalaus klasterių skaičiaus paieška k-medoids algoritmu (Euklido atstumas):
      - empirinis metodas: k ≈ sqrt(n/2) (naudojamas kaip viršutinė k riba),
      - Elbow metodas: k-medoids kaina (atstumų iki medoidų suma),
      - vidutinio silueto metodas.
    Prieš klasterizavimą taikomas Robust Scaling.
    Sukuriami du paveikslai: vidutinis siluetas (k = 2…k_max) ir Elbow (k = 2…k_max ir k = 2…10).
    Papildomai DBSCAN:
      - eps parinkimui nubraižomas k-distance grafikas (žr. plot_k_distance),
        min_samples pagal nutylėjimą = 2 · požymių skaičius;
      - paleidžiamas DBSCAN su dbscan_eps; jei jis nenurodytas – imamas k-distance
        alkūnės eps iš 99 % objektų grafiko.
    """
    X = robust_scale(df, list(df.columns)).values
    n = len(X)
    D = euclidean_matrix(X)

    k_emp = int(round(np.sqrt(n / 2)))
    print(f"Empyrinis metodas: k ≈ sqrt({n}/2) = {k_emp}")
    if k_max is None:
        k_max = k_emp

    if min_samples is None:
        min_samples = 2 * X.shape[1]

    rows = []
    for k in range(2, max(k_max, 10) + 1):
        res = kmedoids.fasterpam(D, k, random_state=random_state)
        rows.append({
            'k': k,
            'cost': res.loss,
            'silhouette': silhouette_score(D, res.labels, metric='precomputed'),
        })
    tab = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/kmedoids_k_lentele.csv", index=False)

    best = {}

    # 1 paveikslas: vidutinis siluetas, k = 2…k_max
    sub = tab[tab['k'] <= k_max]
    k_best = int(sub.loc[sub['silhouette'].idxmax(), 'k'])
    best['silhouette'] = k_best
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub['k'], sub['silhouette'], 'o-', markersize=3)
    ax.axvline(k_best, color='red', linestyle='--', linewidth=1)
    ax.set_title(f"Vidutinio silueto metodas (k-medoids, k = 2…{k_max}, geriausias k={k_best})")
    ax.set_xlabel('Klasterių skaičius k')
    ax.set_ylabel('Vidutinis siluetas')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/kmedoids_siluetas.png", dpi=150)
    plt.close(fig)

    # 2 paveikslas: Elbow, k = 2…k_max ir k = 2…10
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (k_from, k_to) in zip(axes, [(2, k_max), (2, 10)]):
        sub = tab[(tab['k'] >= k_from) & (tab['k'] <= k_to)]
        k_best = elbow_point(sub['k'].values, sub['cost'].values)
        best[f'elbow (k = {k_from}…{k_to})'] = k_best
        ax.plot(sub['k'], sub['cost'], 'o-', markersize=3)
        ax.axvline(k_best, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"k = {k_from}…{k_to} (alkūnė ties k={k_best})")
        ax.set_xlabel('Klasterių skaičius k')
        ax.set_ylabel('Atstumų iki medoidų suma')
        ax.grid(alpha=0.3)
    fig.suptitle("Elbow metodas (k-medoids)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/kmedoids_elbow.png", dpi=150)
    plt.close(fig)

    print(tab.round(4).to_string(index=False))
    print("Geriausi k:", best)

    # DBSCAN: atstumas iki min_samples-ojo kaimyno (pats taškas įskaičiuojamas, kaip sklearn DBSCAN)
    k_dist = np.sort(np.partition(D, min_samples - 1, axis=1)[:, min_samples - 1])
    eps = plot_k_distance(k_dist, min_samples, out_dir=out_dir)
    e = dbscan_eps if dbscan_eps is not None else eps['99%']

    labels = DBSCAN(eps=e, min_samples=min_samples, metric='precomputed').fit_predict(D)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    sizes = np.sort(np.bincount(labels[labels >= 0]))[::-1] if n_clusters else []
    best['DBSCAN'] = {'eps': e, 'klasteriai': n_clusters}
    dbscan_tab = pd.DataFrame([{'eps': e, 'min_samples': min_samples,
                                'klasteriai': n_clusters, 'triukšmas': n_noise,
                                'triukšmo dalis %': round(100 * n_noise / n, 2),
                                'klasterių dydžiai': ', '.join(map(str, sizes))}])
    dbscan_tab.to_csv(f"{out_dir}/dbscan_rezultatai.csv", index=False)
    print("DBSCAN rezultatai:")
    print(dbscan_tab.to_string(index=False))
    return tab, best


def plot_k_distance(k_dist, min_samples, out_dir='exports/k_parinkimas'):
    """
    DBSCAN eps parinkimas k-distance grafiku (Elbow metodas).
    k_dist – atstumų iki min_samples-ojo kaimyno masyvas, surikiuotas didėjimo tvarka.
    eps imamas ties kreivės alkūne – labiausiai po tiese tarp pirmo ir paskutinio taško
    nutolusiu tašku. Kairėje – visi objektai, dešinėje – 99 % objektų
    (be didžiausių atstumų, kurie suspaudžia kreivės pradžią).
    """
    eps = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, frac in zip(axes, [1.0, 0.99]):
        y = k_dist[:int(np.ceil(len(k_dist) * frac))]
        x = np.arange(len(y))
        # Didėjančios kreivės alkūnė yra po tiese tarp pirmo ir paskutinio taško:
        # normuotose koordinatėse imamas taškas, kuriam x - y didžiausias
        xn, yn = x / x[-1], (y - y[0]) / (y[-1] - y[0])
        i_best = int(np.argmax(xn - yn))
        e = float(y[i_best])
        eps[f'{frac:.0%}'] = e
        ax.plot(x, y, linewidth=1.2)
        ax.axhline(e, color='red', linestyle='--', linewidth=1)
        ax.axvline(i_best, color='red', linestyle=':', linewidth=1)
        log = y[-1] / np.median(y) > 20
        if log:
            ax.set_yscale('log')   # kelios didelės išskirtys kitaip suspaudžia visą kreivę
        ax.set_title(f"{frac:.0%} objektų{' (log skalė)' if log else ''} (alkūnė: eps≈{e:.4g})")
        ax.set_xlabel('Objektai (surikiuoti pagal atstumą)')
        ax.set_ylabel(f'Atstumas iki {min_samples}-ojo kaimyno')
        ax.grid(alpha=0.3)
    fig.suptitle(f"DBSCAN eps parinkimas: k-distance grafikas (min_samples={min_samples})")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/dbscan_k_distance.png", dpi=150)
    plt.close(fig)
    print(f"DBSCAN eps (min_samples={min_samples}):", eps)
    return eps


def plot_dendrogram(df, method='single', k_max=15, p=30, out_dir='exports/k_parinkimas'):
    """
    AGNES (aglomeracinio hierarchinio klasterizavimo) dendrograma, Euklido atstumas.
    Prieš klasterizavimą taikomas Robust Scaling.
    Optimalus k parenkamas pagal didžiausią šuolį tarp paskutinių sujungimų aukščių:
    dendrograma kerpama per šio šuolio vidurį.
    Rodomi tik paskutiniai p sujungimai (truncate_mode='lastp'), nes objektų yra 4455.
    """
    X = robust_scale(df, list(df.columns)).values
    Z = linkage(pdist(X, metric='euclidean'), method=method)

    # Sujungimų aukščiai mažėjimo tvarka: tarp h[i-1] ir h[i] kirpimas duoda i+1 klasterių
    h = Z[::-1, 2][:k_max]
    gaps = h[:-1] - h[1:]
    best_i = int(np.argmax(gaps)) + 1
    k_best = best_i + 1
    cut = (h[best_i - 1] + h[best_i]) / 2

    print(f"Dendrograma ({method}): didžiausias šuolis ties k={k_best}, kirpimo aukštis={cut:.4g}")
    print("Paskutinių sujungimų aukščiai:", np.array2string(h, precision=4))

    # Single jungimas sukuria labai gilų (grandininį) medį – scipy piešimui reikia didesnės rekursijos ribos
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    dendrogram(Z, truncate_mode='lastp', p=p, show_contracted=True, ax=ax,
               color_threshold=cut, above_threshold_color='grey')
    ax.axhline(cut, color='red', linestyle='--', linewidth=1,
               label=f"Kirpimas: k={k_best} (aukštis {cut:.4g})")
    ax.set_title(f"AGNES dendrograma ({method} jungimas, paskutiniai {p} sujungimai)")
    ax.set_xlabel("Objektai / sujungtų objektų grupės (skliaustuose – objektų skaičius)")
    ax.set_ylabel("Sujungimo atstumas")
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/agnes_{method}_dendrograma.png", dpi=150)
    plt.close(fig)
    return Z, k_best


# ==============================================================================
# 3. PARAMETRŲ ANALIZĖ IR KLASTERIZAVIMAS PRADINĖJE ERDVĖJE
# ==============================================================================
def draw_clusters(ax, Z, labels, title, xlabel, ylabel, centers=None, outliers=None):
    """
    Nupiešia klasterius dvimatėje projekcijoje Z ant pateiktos ašies ax.
    centers – klasterių centrų (pvz., medoidų) indeksai, jei jie yra.
    outliers – išskirčių kaukė (iš prepare_dataset): išskirtys piešiamos klasterio spalva,
    bet kryželiu ('x'), kiti objektai – tašku.
    """
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    sizes = {int(c): int((labels == c).sum()) for c in clusters}
    print(f"{title} – klasterių dydžiai:", sizes)

    out = np.asarray(outliers, dtype=bool) if outliers is not None else np.zeros(len(labels), bool)
    for i, c in enumerate(clusters):
        color = f"C{i % 10}"
        m = labels == c
        ax.scatter(Z[m & ~out, 0], Z[m & ~out, 1], s=6, alpha=0.6, color=color,
                   label=f"Klasteris {c} (n={sizes[c]})")
        ax.scatter(Z[m & out, 0], Z[m & out, 1], s=14, alpha=0.8, color=color,
                   marker='x', linewidths=0.7)
    if out.any():
        ax.scatter([], [], s=14, color='black', marker='x',
                   label=f"Išskirtys (n={int(out.sum())})")
    if centers is not None:
        ax.scatter(Z[centers, 0], Z[centers, 1], marker='X', s=120,
                   c='black', edgecolors='white', label='Medoidai')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=2)


def plot_panels(panels, suptitle, out_path, ncols=None):
    """
    Išsaugo vieną paveikslą su keliais klasterių grafikais.
    panels – žodynų sąrašas su draw_clusters argumentais (Z, labels, title, xlabel, ylabel, ...).
    ncols – grafikų skaičius eilutėje (pagal nutylėjimą visi vienoje eilutėje).
    """
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6 * nrows), squeeze=False)
    for ax, p in zip(axes.flat, panels):
        draw_clusters(ax, **p)
    for ax in axes.flat[len(panels):]:
        ax.axis('off')
    fig.suptitle(suptitle)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pca_embedding(X, random_state=0):
    """Pirmosios dvi PCA komponentės ir jų ašių pavadinimai (su paaiškinta dispersija)."""
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_ * 100
    print(f"PCA paaiškinta dispersija: PC1={evr[0]:.1f} %, PC2={evr[1]:.1f} %")
    return Z, f"PC1 ({evr[0]:.1f} %)", f"PC2 ({evr[1]:.1f} %)"


def pca_init_from_distances(D, n_components=2):
    """
    PCA tipo t-SNE inicializacija iš atstumų matricos (klasikinis MDS).
    Euklido atstumams tai sutampa su PCA projekcija; sklearn neleidžia init='pca' su
    metric='precomputed', todėl inicializacija apskaičiuojama čia. Mastelis – kaip sklearn
    init='pca' (pirmosios ašies st. nuokrypis 1e-4).
    """
    n = D.shape[0]
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = eigh(B, subset_by_index=[n - n_components, n - 1])
    order = np.argsort(vals)[::-1]
    Y = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
    return (Y / np.std(Y[:, 0]) * 1e-4).astype(np.float32)


def cached_tsne(data, cache_dir, n_components=2, perplexity=30, random_state=0, precomputed=False):
    """
    t-SNE su init='pca' ir rezultatų talpykla (cache).
    data – požymių matrica (precomputed=False) arba atstumų matrica (precomputed=True).
    Raktas – duomenų maišos (hash) reikšmė ir t-SNE parametrai, todėl pasikeitus duomenims
    (pvz., kitai imčiai) ar parametrams t-SNE perskaičiuojamas. Rezultatas saugomas
    {cache_dir}/tsne_<raktas>.npz ir kitą kartą tiesiog nuskaitomas.
    Barnes-Hut t-SNE palaiko tik n_components < 4; didesnei dimensijai naudojamas tikslus
    (lėtas) metodas.
    Grąžina (Z, KL divergencija).
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    h = hashlib.sha1(data.tobytes())
    h.update(f"{data.shape}|{n_components}|{perplexity}|{random_state}|{precomputed}|pca".encode())
    path = os.path.join(cache_dir, f"tsne_{h.hexdigest()[:16]}.npz")
    if os.path.exists(path):
        f = np.load(path)
        print(f"t-SNE (d={n_components}, perplexity={perplexity}) nuskaitytas iš {path}: "
              f"KL divergencija={float(f['kl']):.4f}")
        return f['Z'], float(f['kl'])

    if precomputed:
        init = pca_init_from_distances(data, n_components)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, metric='precomputed',
                    init=init, random_state=random_state,
                    method='barnes_hut' if n_components < 4 else 'exact')
    else:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, init='pca',
                    random_state=random_state,
                    method='barnes_hut' if n_components < 4 else 'exact')
    Z = tsne.fit_transform(data)
    kl = float(tsne.kl_divergence_)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, Z=Z, kl=kl)
    print(f"t-SNE (d={n_components}, perplexity={perplexity}): KL divergencija={kl:.4f}, "
          f"išsaugota {path}")
    return Z, kl


def tsne_embedding(D, perplexity=30, random_state=0,
                   cache_dir='exports/klasterizavimas_pradineje_erdveje/tsne_cache'):
    """
    t-SNE projekcija iš atstumų matricos D (tas pats atstumas, kuriuo klasterizuota),
    init – PCA tipo (klasikinis MDS iš D), rezultatas talpinamas cache_dir.
    t-SNE išlaiko vietinę kaimynystę, bet atstumai tarp klasterių ir jų dydžiai plokštumoje
    nėra interpretuojami.
    """
    Z, _ = cached_tsne(D, cache_dir, 2, perplexity, random_state, precomputed=True)
    return Z


def kmedoids_cluster(df, k, random_state=0, metric='euclidean'):
    """
    k-medoids originalioje (Robust Scaling) erdvėje.
    metric – scipy pdist atstumas ('euclidean' arba 'cityblock' – Manhattan).
    Grąžina X, D ir rezultatą.
    """
    X = robust_scale(df, list(df.columns)).values
    D = squareform(pdist(X, metric=metric))
    return X, D, kmedoids.fasterpam(D, k, random_state=random_state)


def agnes_cluster(df, k, method):
    """AGNES originalioje (Robust Scaling) erdvėje, Euklido atstumas; dendrograma kerpama į k klasterių."""
    X = robust_scale(df, list(df.columns)).values
    d = pdist(X, metric='euclidean')
    Z = linkage(d, method=method)
    labels = fcluster(Z, t=k, criterion='maxclust') - 1   # klasteriai numeruojami nuo 0
    return X, squareform(d), labels


def pair_jaccard_score(labels_true, labels_pred):
    """
    Jaccard indeksas porų atžvilgiu: J = a / (a + b + c), kur
      a – objektų poros, kurios toje pačioje klasėje ir tame pačiame klasteryje,
      b – poros, kurios tame pačiame klasteryje, bet skirtingose klasėse,
      c – poros, kurios toje pačioje klasėje, bet skirtinguose klasteriuose.
    """
    C = pair_confusion_matrix(labels_true, labels_pred)   # elementai skaičiuoja poras du kartus
    a, b, c = C[1, 1], C[0, 1], C[1, 0]
    return float(a / (a + b + c)) if (a + b + c) > 0 else 0.0


def evaluate_clustering(X, labels, y_true, name='', D=None):
    """
    Apskaičiuoja klasterizavimo kokybės matus.
    Vidiniai (naudoja tik duomenis ir klasterius):
      - Silhouette (didesnis geriau, [-1, 1]),
      - Calinski-Harabasz indeksas (didesnis geriau),
      - Davies-Bouldin indeksas (mažesnis geriau).
    Išoriniai (lygina klasterius su tikrosiomis klasėmis y_true):
      - Jaccard indeksas (porų atžvilgiu), ARI, NMI (didesnis geriau).
    X – požymių matrica (Robust Scaling erdvė), D – nebūtina Euklido atstumų matrica
    (jei pateikta, siluetas skaičiuojamas iš jos, kad nereikėtų perskaičiuoti atstumų).
    DBSCAN triukšmo taškai (žymė -1) į vidinius matus neįtraukiami; išoriniuose matuose
    jie paliekami kaip atskira grupė.
    Grąžina žodyną su matais.
    """
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask]))

    res = {'modelis': name, 'klasteriai': n_clusters, 'triukšmas': int((~mask).sum())}
    if n_clusters >= 2 and n_clusters < mask.sum():
        Xm = X[mask]
        if D is not None:
            sil = silhouette_score(D[np.ix_(mask, mask)], labels[mask], metric='precomputed')
        else:
            sil = silhouette_score(Xm, labels[mask], metric='euclidean')
        res['Silhouette'] = sil
        res['Calinski-Harabasz'] = calinski_harabasz_score(Xm, labels[mask])
        res['Davies-Bouldin'] = davies_bouldin_score(Xm, labels[mask])
    else:
        res['Silhouette'] = res['Calinski-Harabasz'] = res['Davies-Bouldin'] = np.nan

    res['Jaccard'] = pair_jaccard_score(y_true, labels)
    res['ARI'] = adjusted_rand_score(y_true, labels)
    res['NMI'] = normalized_mutual_info_score(y_true, labels)

    print(f"{name}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in res.items() if k != 'modelis'))
    return res


METRIC_NAMES = {'euclidean': 'Euklido', 'cityblock': 'Manhattan'}


def metric_table_row(metric_name, res, extra=None):
    """
    k-medoids matų lentelės eilutė: atstumo pavadinimas, papildomi stulpeliai ir matai
    (be modelio aprašo, klasterių ir triukšmo taškų skaičiaus).
    """
    skip = ('modelis', 'klasteriai', 'triukšmas')
    return {'Atstumas': metric_name, **(extra or {}),
            **{key: v for key, v in res.items() if key not in skip}}


def kmedoids_original_space(df, y, k=2, metrics=('euclidean', 'cityblock'), perplexity=30,
                            random_state=0, outliers=None,
                            out_dir='exports/klasterizavimas_pradineje_erdveje'):
    """
    k-medoids klasterizavimas pradinėje (Robust Scaling) erdvėje su skirtingais atstumais.
    Kiekvienam atstumui apskaičiuojami vidiniai ir išoriniai matai.
    Sukuriami du palyginimo paveikslai (po vieną grafiką kiekvienam atstumui):
      - PCA plokštumoje (projekcija ta pati, skiriasi tik klasteriai),
      - t-SNE plokštumoje (t-SNE skaičiuojamas iš to paties atstumo matricos, kuria klasterizuota).
    Matai išsaugomi {out_dir}/kmedoids_k{k}_matai.csv.
    """
    rows, pca_panels, tsne_panels = [], [], []
    Z_pca = None
    for metric in metrics:
        mname = METRIC_NAMES.get(metric, metric)
        X, D, res = kmedoids_cluster(df, k, random_state, metric=metric)
        row = evaluate_clustering(X, res.labels, y, f"k-medoids (k={k}, {mname})", D=D)
        rows.append(metric_table_row(mname, row))
        if Z_pca is None:
            Z_pca, xl, yl = pca_embedding(X, random_state)
        common = dict(labels=res.labels, title=f"{mname} atstumas", centers=res.medoids,
                      outliers=outliers)
        pca_panels.append(dict(Z=Z_pca, xlabel=xl, ylabel=yl, **common))
        tsne_panels.append(dict(Z=tsne_embedding(D, perplexity, random_state),
                                xlabel='t-SNE 1', ylabel='t-SNE 2', **common))

    plot_panels(pca_panels, f"k-medoids klasteriai PCA erdvėje (k={k})",
                f"{out_dir}/kmedoids_k{k}_pca.png")
    plot_panels(tsne_panels, f"k-medoids klasteriai t-SNE erdvėje (k={k}, perplexity={perplexity})",
                f"{out_dir}/kmedoids_k{k}_tsne.png")

    tab = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/kmedoids_k{k}_matai.csv", index=False)
    print(tab.round(4).to_string(index=False))
    return tab


def agnes_original_space(df, y, k=2, methods=('single', 'complete', 'average', 'ward'),
                         perplexity=30, random_state=0, outliers=None,
                         out_dir='exports/klasterizavimas_pradineje_erdveje'):
    """AGNES klasterizavimas pradinėje erdvėje su skirtingais jungimo būdais."""
    X = robust_scale(df, list(df.columns)).values
    d = pdist(X, metric='euclidean')
    D = squareform(d)
    Z_pca, xl, yl = pca_embedding(X, random_state)
    Z_tsne = tsne_embedding(D, perplexity, random_state)

    rows, pca_panels, tsne_panels = [], [], []
    for method in methods:
        Z_link = linkage(d, method=method)
        labels = fcluster(Z_link, t=k, criterion='maxclust') - 1
        rows.append(evaluate_clustering(X, labels, y, f"AGNES (k={k}, {method})", D=D))
        common = dict(labels=labels, title=f"{method} jungimas", outliers=outliers)
        pca_panels.append(dict(Z=Z_pca, xlabel=xl, ylabel=yl, **common))
        tsne_panels.append(dict(Z=Z_tsne, xlabel='t-SNE 1', ylabel='t-SNE 2', **common))

    plot_panels(pca_panels, f"AGNES klasteriai PCA erdvėje (k={k})",
                f"{out_dir}/agnes_k{k}_pca.png")
    plot_panels(tsne_panels, f"AGNES klasteriai t-SNE erdvėje (k={k}, perplexity={perplexity})",
                f"{out_dir}/agnes_k{k}_tsne.png")

    tab = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/agnes_k{k}_matai.csv", index=False)
    print(tab.round(4).to_string(index=False))
    return tab


# ==============================================================================
# 4. KLASTERIŲ STABILUMO EKSPERIMENTAS
# ==============================================================================
def fmt_mean_std(values, digits=2):
    """Vidurkis ± st. nuokrypis lietuvišku formatu, pvz. '0,82 ± 0,06'."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "—"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}".replace('.', ',')


def cluster_stability(df, algorithm='kmedoids', k=2, n_runs=10, frac=0.8,
                      metric='cityblock', method='ward', eps=None, min_samples=None,
                      out_dir='exports/stabilumas'):
    """Klasterių stabilumo eksperimentas originalioje erdvėje (80 % imtys)."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    m = int(round(frac * n))
    samples = [np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))
               for seed in range(n_runs)]

    labelings, run_rows = [], []
    for seed, idx in enumerate(samples):
        df_sub = df.iloc[idx]
        if algorithm == 'kmedoids':
            X, D, res = kmedoids_cluster(df_sub, k, random_state=seed, metric=metric)
            labels = np.asarray(res.labels)
            alg_title = f"k-medoids stabilumas originalioje erdvėje\n(k={k}, {METRIC_NAMES.get(metric, metric)} atstumas"
        elif algorithm == 'agnes':
            X, D, labels = agnes_cluster(df_sub, k, method=method)
            labels = np.asarray(labels)
            alg_title = f"AGNES stabilumas originalioje erdvėje\n(k={k}, {method} jungimas"
        elif algorithm == 'dbscan':
            X = robust_scale(df_sub, list(df_sub.columns)).values
            D = euclidean_matrix(X)
            ms = min_samples if min_samples is not None else 2 * X.shape[1]
            if eps is None:
                k_dist = np.sort(np.partition(D, ms - 1, axis=1)[:, ms - 1])
                y_kd = k_dist[:int(np.ceil(len(k_dist) * 0.99))]
                xn = np.arange(len(y_kd)) / (len(y_kd) - 1)
                yn = (y_kd - y_kd[0]) / (y_kd[-1] - y_kd[0])
                e = float(y_kd[int(np.argmax(xn - yn))])
            else:
                e = eps
            labels = DBSCAN(eps=e, min_samples=ms, metric='precomputed').fit_predict(D)
            alg_title = f"DBSCAN stabilumas originalioje erdvėje\n(min_samples={ms}"
        else:
            raise ValueError(f"Nežinomas algoritmas: {algorithm}")

        labelings.append(pd.Series(labels, index=idx))
        mask = labels != -1
        n_cl = len(np.unique(labels[mask]))
        if n_cl >= 2 and n_cl < mask.sum():
            D_sub = D if mask.all() else D[np.ix_(mask, mask)]
            X_sub = X if mask.all() else X[mask]
            sil = silhouette_score(D_sub, labels[mask], metric='precomputed')
            ch = calinski_harabasz_score(X_sub, labels[mask])
            db = davies_bouldin_score(X_sub, labels[mask])
        else:
            sil = ch = db = np.nan

        run_rows.append({'imtis': seed, 'Silhouette': sil,
                         'Calinski-Harabasz': ch, 'Davies-Bouldin': db})

    pair_rows = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            common = np.intersect1d(samples[i], samples[j])
            li, lj = labelings[i].loc[common].values, labelings[j].loc[common].values
            pair_rows.append({'i': i, 'j': j, 'bendri objektai': len(common),
                              'ARI': adjusted_rand_score(li, lj),
                              'NMI': normalized_mutual_info_score(li, lj),
                              'Jaccard': pair_jaccard_score(li, lj)})

    pairs, runs = pd.DataFrame(pair_rows), pd.DataFrame(run_rows)
    pairs.to_csv(f"{out_dir}/{algorithm}_stabilumas_poros.csv", index=False)
    runs.to_csv(f"{out_dir}/{algorithm}_stabilumas_imtys.csv", index=False)

    ext, internal = ['ARI', 'NMI', 'Jaccard'], ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
    row = {'erdvė': 'Originali'}
    row.update({c: fmt_mean_std(pairs[c]) for c in ext})
    row.update({f"{c} min": f"{pairs[c].min():.2f}".replace('.', ',') for c in ext})
    row.update({c: fmt_mean_std(runs[c]) for c in internal})
    summary = pd.DataFrame([row])
    summary.to_csv(f"{out_dir}/{algorithm}_stabilumas.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([pairs[c] for c in ext], tick_labels=ext)
    ax.set_ylabel('Reikšmė porų (Ci, Cj) bendriems objektams')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f"{alg_title}, {n_runs} imčių po {int(frac * 100)} %, {len(pairs)} poros)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{algorithm}_stabilumas_boxplot.png", dpi=150)
    plt.close(fig)

    kmedoids_stability_compare(pairs, a=compare[0], b=compare[1], n_runs=n_runs, out_dir=out_dir)

    print(summary.to_string(index=False))
    return summary, pairs, runs


def kmedoids_stability(df, k=2, n_runs=10, frac=0.8, metric='cityblock',
                       out_dir='exports/stabilumas'):
    return cluster_stability(df, algorithm='kmedoids', k=k, n_runs=n_runs, frac=frac,
                             metric=metric, out_dir=out_dir)


def agnes_stability(df, k=2, n_runs=10, frac=0.8, method='ward',
                    out_dir='exports/stabilumas'):
    return cluster_stability(df, algorithm='agnes', k=k, n_runs=n_runs, frac=frac,
                             method=method, out_dir=out_dir)


# ==============================================================================
# 5. PARAMETRŲ ANALIZĖ IR KLASTERIZAVIMAS sumažintos dimensijos erdvėje
# ==============================================================================

def pca_dimension_analysis(df, thresholds=(0.8, 0.9), random_state=0,
                           out_dir='exports/klasterizavimas_sumazintoje_erdveje'):
    """
    Sumažintos erdvės dimensijos parinkimas pagal PCA (Robust Scaling duomenims).
    Kriterijai: sukauptos paaiškintos dispersijos slenksčiai (pvz., 80 % ir 90 %)
    ir Kaiserio kriterijus (tikrinė reikšmė > tikrinių reikšmių vidurkis).
    Išsaugo lentelę su kiekvienos komponentės dispersija ir sukauptos dispersijos grafiką.
    Grąžina žodyną {kriterijus: dimensija}.
    """
    os.makedirs(out_dir, exist_ok=True)
    X = robust_scale(df, list(df.columns)).values
    pca = PCA(random_state=random_state).fit(X)
    eig = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    comps = np.arange(1, len(evr) + 1)

    tab = pd.DataFrame({'Komponentė': comps,
                        'Tikrinė reikšmė': np.round(eig, 4),
                        'Paaiškinta dispersija, %': np.round(evr * 100, 2),
                        'Sukaupta dispersija, %': np.round(cum * 100, 2)})
    tab.to_csv(f"{out_dir}/pca_dispersija.csv", index=False)

    dims = {f"{int(t * 100)} % dispersijos": int(np.searchsorted(cum, t) + 1) for t in thresholds}
    dims['Kaiserio kriterijus'] = int((eig > eig.mean()).sum())
    print(f"PCA dimensijos parinkimas (iš viso {len(evr)} požymių):")
    for name, d in dims.items():
        print(f"  {name}: d = {d} (sukaupta dispersija {cum[d - 1] * 100:.1f} %)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(comps, cum * 100, marker='o', ms=4, color='steelblue')
    colors = plt.cm.tab10.colors
    for i, (name, d) in enumerate(dims.items()):
        c = colors[(i + 1) % len(colors)]
        ax.axvline(d, color=c, ls='--', lw=1, label=f"{name}: d = {d}")
        ax.plot(d, cum[d - 1] * 100, 'o', color=c)
    for t in thresholds:
        ax.axhline(t * 100, color='gray', ls=':', lw=1)
    ax.set_xlabel('Komponenčių skaičius')
    ax.set_ylabel('Sukaupta dispersija, %')
    ax.set_title('PCA: sukaupta paaiškinta dispersija')
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(f"{out_dir}/pca_sukaupta_dispersija.png", dpi=150)
    plt.close(fig)
    return dims


def pca_reduce(df, d, random_state=0):
    """
    Robust Scaling + PCA iki d dimensijų.
    Grąžina sumažintos erdvės matricą (n x d) ir kiekvienos komponentės paaiškintą dispersiją
    (dalimis nuo visos originalios erdvės dispersijos).
    """
    X = robust_scale(df, list(df.columns)).values
    pca = PCA(n_components=d, random_state=random_state)
    Xr = pca.fit_transform(X)
    return Xr, pca.explained_variance_ratio_


def tsne_reduce(df, d=2, perplexity=30, random_state=0,
                cache_dir='exports/klasterizavimas_sumazintoje_erdveje/tsne_cache'):
    """
    Robust Scaling + t-SNE (init='pca') iki d dimensijų (Euklido atstumas Robust Scaling erdvėje).
    Rezultatas talpinamas cache_dir: tiems patiems duomenims ir parametrams t-SNE
    skaičiuojamas tik vieną kartą.
    Grąžina sumažintos erdvės matricą (n x d) ir KL divergenciją.
    """
    X = robust_scale(df, list(df.columns)).values
    return cached_tsne(X, cache_dir, d, perplexity, random_state)


REDUCTION_NAMES = {'pca': 'PCA', 'tsne': 't-SNE'}


def kmedoids_reduced_space(df, y, d=12, k=2, method='pca', metrics=('euclidean', 'cityblock'),
                           perplexity=30, random_state=0, outliers=None,
                           out_dir='exports/klasterizavimas_sumazintoje_erdveje'):
    """
    k-medoids klasterizavimas sumažintos dimensijos erdvėje (Robust Scaling + PCA arba t-SNE
    iki d dimensijų). method – 'pca' arba 'tsne' (perplexity naudojamas tik t-SNE).
    Kiekvienam atstumui apskaičiuojami vidiniai ir išoriniai matai (vidiniai – sumažintoje erdvėje;
    t-SNE erdvėje jie dirbtinai išauga, todėl su kitomis erdvėmis jų nelyginti).
    Rezultatai vizualizuojami tik tuo metodu, kuriuo sumažinta dimensija: vienas palyginimo
    paveikslas (po vieną grafiką kiekvienam atstumui):
      - PCA – pirmųjų dviejų tos pačios PCA komponenčių plokštumoje (PC1/PC2),
      - t-SNE – kai d = 2, pats sumažintas žemėlapis; kai d > 2, klasteriai nuspalvinami
        2D t-SNE žemėlapyje iš tų pačių paruoštų duomenų (t-SNE ašys neturi eiliškumo,
        todėl pirmosios dvi d-matės erdvės ašys nieko neparodytų).
    Rezultatai išsaugomi {out_dir}/kmedoids_k{k}_{method}_d{d}.png ir ..._matai.csv.
    """
    mname_red = REDUCTION_NAMES[method]
    if method == 'pca':
        Xr, evr = pca_reduce(df, d, random_state)
        var = float(evr.sum())
        print(f"PCA iki d={d}: paaiškinta dispersija {var * 100:.1f} %")
        # pirmosios dvi komponentės (tos pačios kaip originalios erdvės PCA)
        Z = Xr[:, :2]
        xl, yl = f"PC1 ({evr[0] * 100:.1f} %)", f"PC2 ({evr[1] * 100:.1f} %)"
        extra = {'paaiškinta dispersija, %': round(var * 100, 2)}
        subtitle = f"k={k}, PCA d={d}, {var * 100:.1f} % dispersijos"
    elif method == 'tsne':
        Xr, kl = tsne_reduce(df, d, perplexity, random_state)
        # t-SNE ašys neturi eiliškumo, todėl kai d > 2, klasteriai rodomi 2D t-SNE žemėlapyje,
        # apskaičiuotame iš tų pačių paruoštų duomenų (tas pats žemėlapis kaip d = 2 variante)
        Z = Xr if d == 2 else tsne_reduce(df, 2, perplexity, random_state)[0]
        xl, yl = 't-SNE 1', 't-SNE 2'
        extra = {'KL divergencija': round(kl, 4)}
        subtitle = f"k={k}, t-SNE d={d}, perplexity={perplexity}"
        if d != 2:
            subtitle += ", rodoma 2D t-SNE žemėlapyje"
    else:
        raise ValueError(f"Nežinomas dimensijos mažinimo metodas: {method}")

    rows, panels = [], []
    for metric in metrics:
        mname = METRIC_NAMES.get(metric, metric)
        D = squareform(pdist(Xr, metric=metric))
        res = kmedoids.fasterpam(D, k, random_state=random_state)
        row = evaluate_clustering(Xr, res.labels, y,
                                  f"k-medoids (k={k}, {mname_red} d={d}, {mname})", D=D)
        rows.append(metric_table_row(mname, row, extra))
        panels.append(dict(Z=Z, labels=res.labels, title=f"{mname} atstumas", xlabel=xl, ylabel=yl,
                           centers=res.medoids, outliers=outliers))

    name = f"kmedoids_k{k}_{method}_d{d}"
    plot_panels(panels, f"k-medoids klasteriai sumažintoje erdvėje, {mname_red} vizualizacija "
                        f"({subtitle})", f"{out_dir}/{name}.png")

    tab = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/{name}_matai.csv", index=False)
    print(tab.round(4).to_string(index=False))
    return tab


# k-medoids konfigūracijos palyginimui: (erdvė, dimensija, atstumas); kiekvienai erdvei
# imamas atstumas, kuris buvo geresnis 3 ir 5 skyrių parametrų analizėje
KMEDOIDS_COMPARISON = (('original', None, 'cityblock'),
                       ('pca', 12, 'euclidean'),
                       ('pca', 2, 'cityblock'),
                       ('tsne', 12, 'euclidean'),
                       ('tsne', 2, 'euclidean'))


def kmedoids_comparison(df, y, k=2, configs=KMEDOIDS_COMPARISON, perplexity=30, random_state=0,
                        outliers=None, out_dir='exports/klasterizavimas_sumazintoje_erdveje'):
    """
    k-medoids rezultatų palyginimas pradinėje ir sumažintose erdvėse (kiekvienai erdvei –
    pasirinktas atstumas). Vienas paveikslas su 6 grafikais:
      - viršutinė eilutė – PCA: pradinė erdvė (PCA projekcija), PCA d=12 (PC1/PC2), PCA d=2;
      - apatinė eilutė – t-SNE: pradinė erdvė (t-SNE iš tos pačios atstumų matricos),
        t-SNE d=12 (klasteriai 2D t-SNE žemėlapyje), t-SNE d=2.
    Matai išsaugomi {out_dir}/kmedoids_k{k}_palyginimas_matai.csv (vidiniai matai skaičiuojami
    erdvėje, kurioje klasterizuota, todėl t-SNE erdvių vidiniai matai dirbtinai didesni).
    """
    cache_dir = f"{out_dir}/tsne_cache"
    rows, pca_panels, tsne_panels = [], [], []
    for method, d, metric in configs:
        mname = METRIC_NAMES.get(metric, metric)
        if method == 'original':
            space = 'Pradinė erdvė'
            X, D, res = kmedoids_cluster(df, k, random_state, metric=metric)
        else:
            space = f"{REDUCTION_NAMES[method]} d={d}"
            if method == 'pca':
                X, evr = pca_reduce(df, d, random_state)
            else:
                X, _ = tsne_reduce(df, d, perplexity, random_state, cache_dir=cache_dir)
            D = squareform(pdist(X, metric=metric))
            res = kmedoids.fasterpam(D, k, random_state=random_state)

        row = evaluate_clustering(X, res.labels, y, f"k-medoids (k={k}, {space}, {mname})", D=D)
        rows.append({'Erdvė': space, **metric_table_row(mname, row)})
        common = dict(labels=res.labels, title=f"{space}, {mname} atstumas", centers=res.medoids,
                      outliers=outliers)

        # pradinė erdvė rodoma abiem metodais, sumažinta – tik tuo, kuriuo sumažinta
        if method == 'original':
            Z, xl, yl = pca_embedding(X, random_state)
            pca_panels.append(dict(Z=Z, xlabel=xl, ylabel=yl, **common))
            tsne_panels.append(dict(Z=tsne_embedding(D, perplexity, random_state, cache_dir=cache_dir),
                                    xlabel='t-SNE 1', ylabel='t-SNE 2', **common))
        elif method == 'pca':
            pca_panels.append(dict(Z=X[:, :2], xlabel=f"PC1 ({evr[0] * 100:.1f} %)",
                                   ylabel=f"PC2 ({evr[1] * 100:.1f} %)", **common))
        else:
            # t-SNE ašys neturi eiliškumo: d > 2 atveju klasteriai rodomi 2D t-SNE žemėlapyje
            Z = X if d == 2 else tsne_reduce(df, 2, perplexity, random_state, cache_dir=cache_dir)[0]
            tsne_panels.append(dict(Z=Z, xlabel='t-SNE 1', ylabel='t-SNE 2', **common))

    plot_panels(pca_panels + tsne_panels,
                f"k-medoids (k={k}) palyginimas: viršuje – PCA, apačioje – t-SNE "
                f"(perplexity={perplexity})",
                f"{out_dir}/kmedoids_k{k}_palyginimas.png", ncols=len(pca_panels))

    tab = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(f"{out_dir}/kmedoids_k{k}_palyginimas_matai.csv", index=False)
    print(tab.round(4).to_string(index=False))
    return tab


# ==============================================================================
# 6. SKIRTINGŲ klasterizavimo algoritmų rezultatų palyginimas 
# ==============================================================================


if __name__ == '__main__':

    # 1 skyrius
    df_out, y, outliers = prepare_dataset()
    print(f"Išsaugota exports/duomenys/A28_cleaned.csv: {df_out.shape[0]} eilučių, {df_out.shape[1]} požymių")

    # 2 skyrius
    find_optimal_k(df_out)
    plot_dendrogram(df_out, method='ward')

    # 3 skyrius
    kmedoids_original_space(df_out, y, k=2, outliers=outliers)
    agnes_original_space(df_out, y, k=2, outliers=outliers)

    # 4 skyrius
    kmedoids_stability(df_out, k=2, n_runs=10)
    agnes_stability(df_out, k=2, n_runs=10, method='ward')

    # 5 skyrius
    pca_dims = pca_dimension_analysis(df_out)
    
    kmedoids_reduced_space(df_out, y, d=12, k=2, method='pca', outliers=outliers)
    kmedoids_reduced_space(df_out, y, d=2, k=2, method='pca', outliers=outliers)
    kmedoids_reduced_space(df_out, y, d=12, k=2, method='tsne', perplexity=30, outliers=outliers)
    kmedoids_reduced_space(df_out, y, d=2, k=2, method='tsne', perplexity=30, outliers=outliers)
    kmedoids_comparison(df_out, y, k=2, outliers=outliers)

    # 6 skyrius

