import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def graph_housing(data: pd.DataFrame, city: str) -> plt.Figure:
    data_format = pd.melt(
                          data[["LIB_MOD","proportion_city","proportion_national"]].rename(columns={"proportion_city": city,
                                                                                                    "proportion_national": "France entière"}),
                          id_vars = ["LIB_MOD"], value_vars = [city,"France entière"]
                          )
    fig, ax = plt.subplots(1, figsize=(11, 6))
    ax = sns.barplot(data = data_format, x="value", y ="LIB_MOD", hue="variable", palette="mako", orient='h')
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=True, offset=None, trim=True)
    score_percent_city = round(data_format.query("variable == @city")["value"] * 100,1).astype(str) + "%"
    score_percent_nat = round(data_format.query("variable == 'France entière'")["value"] * 100,1).astype(str) + "%"
    try:
        ax.bar_label(ax.containers[0], labels = score_percent_city,fmt='%.f')
        ax.bar_label(ax.containers[1], labels = score_percent_nat,fmt='%.f')
    except Exception:
        pass

    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_facecolor('white')
    ax.set(xticklabels=[])
    ax.legend().set_title(None)
    return fig

def graph_pluvio(data_pluvio: pd.DataFrame, city: str) -> plt.Figure:
    data_format = pd.melt(data_pluvio.reset_index(), id_vars = ["month_name"], value_vars = [city,"Moy. villes françaises"])
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax = sns.barplot(data = data_format, x="month_name", y ="value", hue="variable", palette="mako")
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax.legend().set_title(None)
    ax.set_ylabel("mm")
    ax.set_xlabel(None)
    sns.set_style("darkgrid")
    return fig

def graph_poll(data_poll: pd.DataFrame, city: str, reco_OMS: float) -> plt.Figure:
    data_format = pd.melt(data_poll.reset_index(), id_vars = ["date_clean"], value_vars = [city,"Moy. nationale", "Recommandation OMS"])
    fig, ax = plt.subplots(1, figsize=(12, 4))
    ax = sns.lineplot(data = data_format.query("variable != 'Recommandation OMS'"), x="date_clean", y ="value", hue="variable", palette="mako")
    ax = sns.lineplot(data = data_format.query("variable == 'Recommandation OMS'"), x="date_clean", y ="value", color="red", linestyle='--')
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax.legend().set_title(None)
    ax.set_ylabel("µg/m3")
    ax.set_xlabel(None)
    ax.annotate("Seuil journalier recommandé par l'OMS", xy = (pd.to_datetime('2023-01-15'), reco_OMS), xytext=(pd.to_datetime('2023-01-04'), reco_OMS + 0.5), color='red')
    return fig
