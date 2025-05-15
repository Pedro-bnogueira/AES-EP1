
"""SmartRefactor vs. Refatoração Tradicional
=============================================
Script de análise estatística reproduzível para o experimento de avaliação
de ferramentas de refatoração (SmartRefactor x Eclipse Tradicional).

Análises:
    • Integração do perfil dos participantes
    • Estatísticas descritivas do perfil
    • Testes de influência do perfil sobre erros funcionais e problemas de design

Saídas geradas:
    - descriptive_stats.csv
    - hypothesis_results.csv
    - profile_stats.csv
    - profile_influence.csv
    - boxplot_<metrica>.png
    - corr_scatter_experience_vs_errors.png (exemplo)
Dependências:
    Python >= 3.9
    pandas, scipy, matplotlib
"""

from pathlib import Path
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# Configuração de caminhos
metrics_csv = Path("./coleta.csv")
profile_csv = Path("./perfil_dos_participantes.csv")
out_dir = Path("./results")
out_dir.mkdir(parents=True, exist_ok=True)

def load_metrics(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    rename = {
        "Tempo (h)": "tempo_h",
        "LOC Modificadas": "loc_mod",
        "Erros Funcionais": "erros",
        "Problemas de Design": "design",
        "Ferramenta": "ferramenta",
    }
    df = df_raw.rename(columns=rename)
    for col in ["tempo_h", "loc_mod"]:
        df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False).astype(float)
    df["erros"] = pd.to_numeric(df["erros"], errors="coerce")
    df["design"] = pd.to_numeric(df["design"], errors="coerce")
    # remove colunas extras não numéricas
    df = df[["ID", "tempo_h", "erros", "design", "ferramenta"]]
    return df

def load_profile(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normaliza nomes de coluna
    rename = {
        "Formacao": "formacao",
        "Experiencia": "experiencia",
        "Conhecimento_Refatoracao": "kn_ref",
        "Conhecimento_Java": "kn_java"
    }
    df = df.rename(columns=rename)
    return df

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    desc = (
        df.groupby("ferramenta")
        .agg(
            n=("tempo_h", "count"),
            tempo_medio=("tempo_h", "mean"),
            tempo_dp=("tempo_h", "std"),
            erros_medio=("erros", "mean"),
            erros_dp=("erros", "std"),
            design_medio=("design", "mean"),
            design_dp=("design", "std"),
        )
        .round(3)
    )
    return desc

def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for metric, label in [
        ("tempo_h", "Tempo (h)"),
        ("erros", "Erros Funcionais"),
        ("design", "Problemas de Design"),
    ]:
        grp_smart = df[df["ferramenta"] == "SmartRefactor"][metric]
        grp_trad = df[df["ferramenta"] == "Tradicional"][metric]

        p_smart = stats.shapiro(grp_smart).pvalue
        p_trad = stats.shapiro(grp_trad).pvalue
        normal = (p_smart > 0.05) and (p_trad > 0.05)

        if normal:
            stat, pval = stats.ttest_ind(grp_smart, grp_trad, equal_var=False)
            test_name = "t (Welch)"
            effect = abs(grp_smart.mean() - grp_trad.mean()) / math.sqrt((grp_smart.var(ddof=1)+grp_trad.var(ddof=1))/2)
        else:
            stat, pval = stats.mannwhitneyu(grp_smart, grp_trad, alternative="two-sided")
            test_name = "Mann–Whitney"
            z = stats.norm.isf(pval/2)
            effect = abs(z) / (len(grp_smart)+len(grp_trad))**0.5

        results.append({
            "Métrica": label,
            "Teste": test_name,
            "Estatística": round(stat,3),
            "p_valor": round(pval,5),
            "Normal?": normal,
            "Efeito": round(effect,3)
        })
    return pd.DataFrame(results)

def profile_descriptive(df_profile: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    for col in ["formacao", "experiencia", "kn_ref", "kn_java"]:
        counts[col] = df_profile[col].value_counts().to_dict()
    return pd.DataFrame(counts)

def map_ordinal(series: pd.Series, order):
    return series.map({cat: idx for idx, cat in enumerate(order)}).astype(float)

def profile_influence(df: pd.DataFrame):
    # Ordem para mapeamento ordinal
    exp_order = ["0-1 anos","1-2 anos","3-5 anos","6+ anos"]
    kn_order  = ["Nenhum","Básico","Razoável","Avançado"]

    df["exp_num"] = map_ordinal(df["experiencia"], exp_order)
    df["kn_ref_num"] = map_ordinal(df["kn_ref"], kn_order)
    df["kn_java_num"] = map_ordinal(df["kn_java"], kn_order)

    results = []

    # Correlação Spearman (ordinal vs métrica)
    for ordinal_col, label in [("exp_num", "Experiência"),
                               ("kn_ref_num", "Conhecimento Refatoração"),
                               ("kn_java_num", "Conhecimento Java")]:
        for metric, metric_label in [("erros","Erros"), ("design","Design")]:
            coef, pval = stats.spearmanr(df[ordinal_col], df[metric], nan_policy="omit")
            results.append({
                "Variável ordinal": label,
                "Métrica": metric_label,
                "Teste": "Spearman ρ",
                "Coeficiente": round(coef,3),
                "p_valor": round(pval,5)
            })

    # Kruskal‑Wallis experiência vs erros/design
    for metric, metric_label in [("erros","Erros"), ("design","Design")]:
        groups = [df[df["experiencia"]==lvl][metric].dropna() for lvl in exp_order if not df[df["experiencia"]==lvl].empty]
        if len(groups) >= 2:
            stat, pval = stats.kruskal(*groups)
            results.append({
                "Variável ordinal": "Experiência (categorias)",
                "Métrica": metric_label,
                "Teste": "Kruskal‑Wallis",
                "Coeficiente": round(stat,3),
                "p_valor": round(pval,5)
            })
    return pd.DataFrame(results)

def create_boxplots(df: pd.DataFrame):
    for metric, label in [
        ("tempo_h", "Tempo (h)"),
        ("erros", "Erros Funcionais"),
        ("design", "Problemas de Design")]:
        plt.figure()
        df.boxplot(column=metric, by="ferramenta")
        plt.title(f"Distribuição de {label} por ferramenta")
        plt.suptitle("")
        plt.xlabel("Ferramenta")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(out_dir / f"boxplot_{metric}.png", dpi=300)
        plt.close()

# -------------------------------------------------------------------------#
def main():
    # Carrega dados principais e perfil
    df_metrics = load_metrics(metrics_csv)
    df_profile  = load_profile(profile_csv)

    # Mescla pelo ID
    df = pd.merge(df_metrics, df_profile, on="ID", how="left")

    # Estatísticas descritiva
    desc = descriptive_stats(df)
    desc.to_csv(out_dir / "descriptive_stats.csv")
    print("Descritiva salva em", out_dir / "descriptive_stats.csv")

    # Testes de hipótese
    htest = hypothesis_tests(df)
    htest.to_csv(out_dir / "hypothesis_results.csv", index=False)
    print("Hipóteses salvas em", out_dir / "hypothesis_results.csv")

    # Estatísticas do perfil 
    pdesc = profile_descriptive(df_profile)
    pdesc.to_csv(out_dir / "profile_stats.csv")
    print("Perfil descritivo salvo em", out_dir / "profile_stats.csv")

    pinf = profile_influence(df)
    pinf.to_csv(out_dir / "profile_influence.csv", index=False)
    print("Influencia de perfil salva em", out_dir / "profile_influence.csv")

    # Gráficos Boxplot
    create_boxplots(df)
    print("Boxplots gerados em", out_dir)

if __name__ == "__main__":
    main()