# AGENTS.md

## Project overview

This repository contains a single Python script, [lab3_tyrimas.py](lab3_tyrimas.py), written for a
**machine learning course assignment (lab 3: clustering research)**. It clusters the `A28.csv` dataset
(K-Medoids, AGNES, DBSCAN), chooses the number of clusters, evaluates the results and exports plots
and tables. Comments, printed output and file names are in Lithuanian — keep it that way.

Run it with:

```bash
python lab3_tyrimas.py
```

## Code structure — sections

The script is divided into numbered sections marked by banner comments:

```python
# ==============================================================================
# N. SECTION TITLE
# ==============================================================================
```

Current sections:

1. `DUOMENŲ PARUOŠIMAS` — data cleaning (duplicates, missing values, outliers, Spearman correlation filter)
2. `OPTIMALAUS KLASTERIŲ SKAIČIAUS NUSTATYMAS` — choosing k (empirical, Elbow, silhouette, k-distance, dendrograms)
3. `PARAMETRŲ ANALIZĖ IR KLASTERIZAVIMAS PRADINĖJE ERDVĖJE` — clustering in the original feature space
4. `KLASTERIŲ STABILUMO EKSPERIMENTAS` — cluster stability experiment
5. `PARAMETRŲ ANALIZĖ IR KLASTERIZAVIMAS sumažintos dimensijos erdvėje` — clustering in reduced-dimension space
6. `SKIRTINGŲ klasterizavimo algoritmų rezultatų palyginimas` — comparison of clustering algorithms

Rules:

- **Put every new function into the section it logically belongs to.** Do not append code at the end
  of the file or create catch-all helper areas. A helper used by several sections goes into the
  earliest section that uses it.
- Keep the banner comment format and numbering. Add a new section only if the work does not fit
  any existing one.
- The `if __name__ == '__main__':` block is also divided by section comments (`# 1 skyrius`,
  `# 2 skyrius`, …). Calls for each section go under the matching comment, in section order.
  Keep this structure when adding or changing calls.

## Exports

All outputs go under `exports/`, with **one folder per section**. Each section's functions write only
to their own folder (passed as an `out_dir` / `out_path` default argument):

| Section | Folder |
|---|---|
| 1 | `exports/duomenys/` |
| 2 | `exports/k_parinkimas/` |
| 3 | `exports/klasterizavimas_pradineje_erdveje/` |
| 4 | new folder for the stability experiment (e.g. `exports/stabilumas/`) |
| 5 | new folder for reduced-dimension clustering (e.g. `exports/klasterizavimas_sumazintoje_erdveje/`) |
| 6 | new folder for the algorithm comparison (e.g. `exports/palyginimas/`) |

When starting a new section, create its folder (`os.makedirs(out_dir, exist_ok=True)`) and use it
instead of reusing another section's folder. Note: some section 3 plotting helpers still default to
`exports/vizualizacija/`; output produced for a section should end up in that section's folder.

## Fixed choices

- **Number of clusters: k = 2.** Section 2 settled on this value; use it in all later sections unless the
  task explicitly asks to vary k.
- **Visualisation uses exactly two dimensionality-reduction methods:**
  - **PCA** (2 components): `pca_embedding` / `plot_clusters_pca`
  - **t-SNE with perplexity = 30**: `tsne_embedding` / `plot_clusters_tsne`

  Which of them to use depends on the space the clustering was done in:
  - **Original space** (sections 3, 4 and the original-space side of 6): show every result
    with **both** PCA and t-SNE.
  - **Reduced space** (section 5 and the reduced-space side of 6): show the result **only with the
    method that was used to reduce the dimension**, in that method's own projection:
    - reduced with PCA → plot the first two components of that same PCA (no t-SNE figure);
    - reduced with t-SNE → plot the t-SNE map itself (no PCA figure).

    Do not apply a second reduction method to already reduced data (e.g. PCA of a t-SNE map).

## Comparing parameters in one figure

When researching different parameter values (distance metric, linkage method, eps, etc.), put all
parameter variants side by side **in one figure per visualisation method** (use `plot_panels`),
not in a separate file per variant.

Example: `kmedoids_original_space` compares the Euclidean and Manhattan (`cityblock`) metrics with
exactly 2 pictures:

- `kmedoids_k2_pca.png`: Euclidean | Manhattan, both shown with PCA
- `kmedoids_k2_tsne.png`: Euclidean | Manhattan, both shown with t-SNE

The metrics for all variants go into one CSV table (e.g. `kmedoids_k2_matai.csv`). Follow this
pattern for new parameter studies.

## Robust Scaling — always apply it

`prepare_dataset()` saves and returns the cleaned data **without scaling** (`exports/duomenys/A28_cleaned.csv`
is unscaled). This is intentional — the stability experiment (section 4) takes 80% subsamples, and
scaling must be computed on the data actually used.

Therefore **every function that does anything with the data (clustering, distances, PCA, t-SNE,
metrics, k selection, etc.) must apply Robust Scaling first**, using the existing helper:

```python
X = robust_scale(df, list(df.columns)).values
```

- Apply it to the exact subset being analysed (e.g. each 80% subsample in section 4), not once globally.
- Never pass the raw `df_out` from `prepare_dataset()` straight into an algorithm or metric.
- Do not include the `class` column (`y`) in the scaled features — it is only used for evaluation
  against the clusters.
- Use `robust_scale` (median / IQR) rather than other scalers, so all sections stay comparable.
