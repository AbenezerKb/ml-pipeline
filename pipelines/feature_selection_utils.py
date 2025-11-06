from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def identify_features_to_drop(
    df: pd.DataFrame,
    X_train_preprocess: pd.DataFrame,
    iv_scores_series: pd.Series,
    mutual_info: pd.Series,
    threshold_corr: float = 0.75,
    threshold_iv: float = 1e-4,
    threshold_mi: float = 1e-3,
    nan_features: Optional[List[str]] = None,
) -> List[str]:
    """
    Identifie les features à supprimer selon plusieurs critères :
      - forte corrélation entre features numériques (corr > threshold_corr)
      - IV (information value) plus faible que threshold_iv
      - mutual information plus faible que threshold_mi
      - colonnes listées dans nan_features (par défaut ["HandsetPrice"])

    Retour:
      - liste de noms de colonnes à considérer pour suppression (liste unique)
    """

    print("\nIDENTIFYING FEATURES TO CONSIDER FOR REMOVAL...")
    print("=" * 100)

    # --- 1) Corrélation élevée (numerical columns intersection) ---
    numerical_cols_processed = df.select_dtypes(include=["float64", "int64"]).columns.intersection(
        X_train_preprocess.columns
    )
    if len(numerical_cols_processed) > 0:
        corr_matrix = X_train_preprocess[numerical_cols_processed].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_highly_correlated = [
            col for col in upper.columns if any(upper[col] > threshold_corr)
        ]
    else:
        to_drop_highly_correlated = []

    print(f"Identified {len(to_drop_highly_correlated)} features with high correlation (>{threshold_corr})")
    print("   →", to_drop_highly_correlated)
    print("=" * 100)

    # --- 2) Faible IV ---
    if iv_scores_series is None or not isinstance(iv_scores_series, pd.Series):
        to_drop_low_iv = []
    else:
        to_drop_low_iv = iv_scores_series[iv_scores_series < threshold_iv].index.tolist()

    print(f"Identified {len(to_drop_low_iv)} features with low IV (<{threshold_iv})")
    print("   →", to_drop_low_iv)
    print("=" * 100)

    # --- 3) Faible Mutual Information ---
    if mutual_info is None or not isinstance(mutual_info, pd.Series):
        to_drop_low_mi = []
    else:
        to_drop_low_mi = mutual_info[mutual_info < threshold_mi].index.tolist()

    print(f" Identified {len(to_drop_low_mi)} features with low MI (<{threshold_mi})")
    print("   →", to_drop_low_mi)
    print("=" * 100)

    # --- 4) Colonnes à supprimer pour NaN (par défaut HandsetPrice) ---
    if nan_features is None:
        nan_features = ["HandsetPrice"]
    # on ne garde que celles qui existent dans df
    nan_features_existing = [c for c in nan_features if c in df.columns]

    print(f" Identified {len(nan_features_existing)} features due to missing-values rule")
    print("   →", nan_features_existing)
    print("=" * 100)

    # --- 5) Fusionner et renvoyer (liste unique) ---
    to_drop = list(
        set(to_drop_highly_correlated + to_drop_low_iv + to_drop_low_mi + nan_features_existing)
    )

    print(f"Total distinct features to consider dropping: {len(to_drop)}")
    print("   →", to_drop)
    print("=" * 100)

    return to_drop


def drop_features_from_df(
    df: pd.DataFrame,
    features_to_drop: List[str],
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Supprime les colonnes listées dans features_to_drop du DataFrame `df`.
    - inplace: si True, modifie df, sinon retourne une copie modifiée
    - verbose: si True, affiche un résumé

    Retourne le DataFrame modifié.
    """
    if not inplace:
        df = df.copy()

    present = [c for c in features_to_drop if c in df.columns]
    missing = [c for c in features_to_drop if c not in df.columns]

    if present:
        df.drop(columns=present, inplace=True)
    if verbose:
        print(f"Dropped {len(present)} columns: {present}")
        if missing:
            print(f"{len(missing)} requested columns not present in df and were ignored: {missing}")

    return df


def identify_and_drop(
    df: pd.DataFrame,
    X_train_preprocess: pd.DataFrame,
    iv_scores_series: pd.Series,
    mutual_info: pd.Series,
    *,
    threshold_corr: float = 0.75,
    threshold_iv: float = 1e-4,
    threshold_mi: float = 1e-3,
    nan_features: Optional[List[str]] = None,
    inplace: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combinaison utilitaire : identifie les colonnes à supprimer puis les supprime.
    Retourne (df_modifié, liste_features_supprimées)
    """
    to_drop = identify_features_to_drop(
        df=df,
        X_train_preprocess=X_train_preprocess,
        iv_scores_series=iv_scores_series,
        mutual_info=mutual_info,
        threshold_corr=threshold_corr,
        threshold_iv=threshold_iv,
        threshold_mi=threshold_mi,
        nan_features=nan_features,
    )

    df_after = drop_features_from_df(df if inplace else df.copy(), to_drop, inplace=inplace, verbose=verbose)
    return df_after, to_drop
