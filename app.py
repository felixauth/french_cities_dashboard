import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PIL import Image
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.express as px
import geopy.distance
import os
import calendar

#################################### FUNCTIONS #######################################################

@st.cache_data
def load_data(file_path:str):
    df = pd.read_parquet(file_path)
    return df

def graph_housing(data, city):
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
    ax.bar_label(ax.containers[0], labels = score_percent_city,fmt='%.f')
    ax.bar_label(ax.containers[1], labels = score_percent_nat,fmt='%.f')

    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_facecolor('white')
    ax.set(xticklabels=[])
    ax.legend().set_title(None)

    return fig

def dvf_preproc(dvf_clean_df):

    """
    Returns a dataframe with real estate transactions for houses and apartments, cleaned of outliers and duplicates.
    """
    dvf_house_apt = dvf_clean_df.loc[
        (dvf_clean_df["Type local"].isin(["Maison", "Appartement"])) &
        (dvf_clean_df["Nature mutation"] == "Vente")
        ].reset_index(drop=True)

    # Excluding outliers
    dvf_preproc_unique = dvf_house_apt.drop_duplicates(subset=["Date mutation","Valeur fonciere", "No voie", "Voie"])
    lower_threshold = dvf_preproc_unique["Valeur fonc / surface habitable"].quantile(.05)
    upper_threshold = dvf_preproc_unique["Valeur fonc / surface habitable"].quantile(.95)
    dvf_preproc_clean = dvf_preproc_unique.copy()
    dvf_preproc_clean["Valeur fonc / surface habitable"] = dvf_preproc_clean["Valeur fonc / surface habitable"].clip(lower=lower_threshold, upper=upper_threshold)
    dvf_preproc_no_outlier = dvf_preproc_clean[~dvf_preproc_clean["Valeur fonc / surface habitable"].isin([lower_threshold, upper_threshold])]

    return dvf_preproc_no_outlier

def dvf_per_city(dvf_preproc):

    dvf_house_apt_avg = dvf_preproc.groupby(['CODGEO','Type local'], as_index = False)["Valeur fonc / surface habitable"].mean().sort_values(by=["CODGEO","Type local"]).reset_index(drop=True)

    return dvf_house_apt_avg

def data_meteo_national_avg(data_meteo_preproc_df):

    data_nat = data_meteo_preproc_df.groupby("date_clean")[["precipitations_3h","humidite_%","temperature_C"]].mean()

    return data_nat

@st.cache_data
def wheather_station_list(data_meteo_preproc_df):

    stations_coord = data_meteo_preproc_df.query("numer_sta != 7661").loc[:,["codegeo","nom_epci","libgeo","numer_sta","latitude","longitude"]].drop_duplicates(subset=["numer_sta"]).sort_values(by=["libgeo"]).reset_index(drop=True)
    stations_coord["coord"] = list(zip(stations_coord["latitude"], stations_coord["longitude"]))
    return stations_coord

@st.cache_data
def pollution_station_list(data_pollution_df):

    stations_coord = data_pollution_df.loc[:,["Zas","code site","nom site","coord"]]\
        .drop_duplicates(subset=["code site"])\
            .sort_values(by=["nom site"])\
                .reset_index(drop=True)\
                    .rename(columns={"code site":"numer_sta"})

    return stations_coord

def find_closest_station(station_df, coordinates: tuple):

    closest_station = station_df.copy()
    closest_station["distance"] = closest_station.apply(lambda x: geopy.distance.geodesic(x["coord"], coordinates).km, axis=1)

    return closest_station[closest_station["distance"] == closest_station["distance"].min()].loc[:,"numer_sta"].values[0]


def temp_by_season(data_meteo_city):
    df = data_meteo_city.copy()
    df["Saison"] = np.where(
        df["date_clean"].dt.month.isin([3, 4, 5]),
        "printemps",
        np.where(
            df["date_clean"].dt.month.isin([6, 7, 8]),
            "été",
            np.where(
                df["date_clean"].dt.month.isin([9, 10, 11]),
                "automne",
                "hiver"
        )))

    df_grouped = df.groupby(["Saison"])["temperature_C"].mean()

    return df_grouped

@st.cache_data
def data_meteo_national_avg(data_meteo_preproc_df):

    data_nat = data_meteo_preproc_df.dropna(subset=["precipitations_3h","humidite_%","temperature_C"]).groupby("date_clean")[["precipitations_3h","humidite_%","temperature_C"]].mean()

    return data_nat

def pluvio_moyenne(data_preproc):
    df = data_preproc.copy().reset_index()
    df["month_name"] = df["date_clean"].dt.month.apply(lambda x : calendar.month_name[x])   #dt.month_name("fr_FR.utf8")
    df["year_month"] = df["date_clean"].dt.to_period("M")
    #sum of rain per month
    df_sum_month = df.groupby(["year_month", "month_name"], as_index=False)["precipitations_3h"].sum()

    #average of rain per month
    df_sum_month["month"] = df_sum_month["year_month"].dt.month
    df_avg_month = df_sum_month.groupby(["month_name","month"], as_index=False)["precipitations_3h"].mean().sort_values(by=["month"]).reset_index(drop=True)
    return df_avg_month

def graph_pluvio(data_pluvio, city):
    data_format = pd.melt(data_pluvio.reset_index(), id_vars = ["month_name"], value_vars = [city,"Moy. villes françaises"])
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax = sns.barplot(data = data_format, x="month_name", y ="value", hue="variable", palette="mako")
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax.legend().set_title(None)
    ax.set_ylabel("mm")
    ax.set_xlabel(None)
    sns.set_style("darkgrid")
    return fig

def graph_poll(data_poll, city, reco_OMS):
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

def replace_checkmark(data: pd.Series):
     data_checkmark = data.replace(1,"✔️").replace(0,"❌")
     return data_checkmark

#################################### WELCOME PAGE #######################################################

st.set_page_config(
    page_title="City_Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

folder = "processed_data"

title_image, title_text = st.columns((1,4))
with title_image:
    image_france = Image.open('processed_data/logo-france-2.png')
    st.image(image_france, width=120)
with title_text:
    st.markdown('# *Explorateur de villes :blue[françaises]*')


folder = "processed_data"
insee_data_file = os.path.join(folder,"2020_insee_data_population.parquet.gzip")
insee_data_sum_stat = os.path.join(folder,"2020_insee_data_population_sum_stat.parquet.gzip")
cityhall_elec_data_file = os.path.join(folder,"2020_gouv_data_city_hall_elections.parquet.gzip")
pres_elec_data_file = os.path.join(folder,"2022_gouv_data_pres_elections.parquet.gzip")
insee_city_name_id = os.path.join(folder,"cities_insee_id.parquet.gzip")
insee_housing_const = os.path.join(folder,"2020_housing_constr_period.parquet.gzip")
insee_hlm = os.path.join(folder,"2020_housing_hlm.parquet.gzip")
insee_immigration = os.path.join(folder,"2020_housing_immigration.parquet.gzip")
insee_nb_rooms = os.path.join(folder,"2020_housing_nb_rooms.parquet.gzip")
insee_owner_share = os.path.join(folder,"2020_housing_owner_share.parquet.gzip")
insee_housing_size = os.path.join(folder,"2020_housing_size.parquet.gzip")
insee_housing_age = os.path.join(folder,"2020_housing_age.parquet.gzip")
real_estate_2023 = os.path.join(folder,"2023_real_estate_mkt.parquet.gzip")
wheather_data_2020_2023 = os.path.join(folder,"2020_2023_wheather.parquet.gzip")
pollution_data_2023 = os.path.join(folder,"2023_pollution.parquet.gzip")
local_tax_2022 = os.path.join(folder, "2022_local_taxes.parquet.gzip")
rental_2022 = os.path.join(folder, "2022_rental_indices.parquet.gzip")
schools = os.path.join(folder, "schools_data.parquet.gzip")

insee_city_name_df = load_data(insee_city_name_id)

list_cities = list(
     insee_city_name_df[
          ~insee_city_name_df["LIBELLE_COM"].isin(["Paris","Marseille","Lyon"])
          ]["LIB_DEP"].unique()
          )

# Selecting the city
city = st.selectbox(
    'Sélectionner la ville :',
    list_cities,
    index=None,
    placeholder="Sélectionner une ville dans la liste déroulante"
    )

launch_button = st.button("Afficher les données", type="primary")

if launch_button:

    if city is None:

        st.text('⚠️ Veuillez sélectionner une ville !')

    else:

        with st.spinner(text="Chargement des données..."):

            #################################### LOADING DATA #######################################################

            #Geographic data on selected city
            insee_city_id = insee_city_name_df[insee_city_name_df["LIB_DEP"] == city]["COM"].iloc[0].lstrip("0")
            insee_city_dep_name = insee_city_name_df[insee_city_name_df["LIB_DEP"]== city]["LIBELLE_DEP"].iloc[0]
            insee_city_dep_num = insee_city_name_df[insee_city_name_df["LIB_DEP"]== city]["DEP"].iloc[0]
            insee_city_reg = insee_city_name_df[insee_city_name_df["LIB_DEP"]== city]["LIBELLE_REG"].iloc[0]

            st.title(f"📌 {city}")
            st.markdown(f"🗺️ *{insee_city_dep_name} ({insee_city_dep_num}) - {insee_city_reg}*")
            st.markdown("""___""")

            #Importing summart stats on insee data
            insee_sum_stats = load_data(insee_data_sum_stat)
            #Importing data on selected city
            criteria = [("CODGEO", "==", insee_city_id)]
            data_combined_enriched = pd.read_parquet(insee_data_file, filters = criteria)
            cityhall_elec_data = pd.read_parquet(cityhall_elec_data_file, filters = criteria).iloc[0]
            pres_elec_data = pd.read_parquet(pres_elec_data_file, filters = criteria)
            insee_hlm_data = pd.read_parquet(insee_hlm, filters = criteria)
            insee_immigration_data = pd.read_parquet(insee_immigration, filters = criteria)
            insee_owner_share_data = pd.read_parquet(insee_owner_share, filters = criteria)
            insee_age_pop = pd.read_parquet(insee_housing_age, filters = criteria)
            insee_surface = pd.read_parquet(insee_housing_size, filters = criteria)
            dvf_2023 = pd.read_parquet(real_estate_2023, filters = criteria)
            taxes = pd.read_parquet(local_tax_2022)
            rental = pd.read_parquet(rental_2022, filters=criteria)
            schools_data = pd.read_parquet(schools, filters=criteria)

            #################################### STORING VALUES #######################################################

            #Importing data on wheather station
            data_meteo_preproc_df = pd.read_parquet(wheather_data_2020_2023)

            #Importing data on pollution
            data_pollution_df = pd.read_parquet(pollution_data_2023)

            #Insee data - Getting the national values for comparison
            pop_median = insee_sum_stats.loc["Median","P20_POP"]
            poverty_rate_median = insee_sum_stats.loc["Median","TP6020"]
            standard_of_living_median = insee_sum_stats.loc["Median","MED20"]
            principal_residency_median = insee_sum_stats.loc["Median","RP_SHARE"]
            unemployment_rate_median = insee_sum_stats.loc["Median","CHOM_RATE"]
            empty_residency_median = insee_sum_stats.loc["Median","LOGVAC_RATE"]

            #Insee data - Getting the city values
            # data_city = data_combined_enriched[data_combined_enriched["CODGEO"] == insee_city_id]
            pop_city = data_combined_enriched["P20_POP"].values[0]
            pop_growth_city = data_combined_enriched["POP_GROWTH_RATE"].values[0]
            principal_residency_city = data_combined_enriched["RP_SHARE"].values[0]
            unemployment_rate_city = data_combined_enriched["CHOM_RATE"].values[0]
            empty_residency_city = data_combined_enriched["LOGVAC_RATE"].values[0]
            city_lat = float(data_combined_enriched["latitude"].values[0])
            city_long  = float(data_combined_enriched["longitude"].values[0])
            poverty_rate = float(data_combined_enriched["TP6020"].values[0])
            standard_of_living = float(data_combined_enriched["MED20"].values[0])

            try:
                hlm_share = insee_hlm_data["proportion_city"].values[0]
                hlm_share_nat = insee_hlm_data["proportion_national"].values[0]
            except IndexError:
                hlm_share = None
                hlm_share_nat = None
            try:
                immigration_share_nat = insee_immigration_data.query("IMMIM == '1'")["proportion_national"].values[0]
                immigration_share = insee_immigration_data.query("IMMIM == '1'")["proportion_city"].values[0]
            except IndexError:
                immigration_share_nat = None
                immigration_share = None
            try:
                owner_share_nat = insee_owner_share_data["proportion_national"].values[0]
                owner_share = insee_owner_share_data["proportion_city"].values[0]
            except IndexError:
                owner_share_nat = None
                owner_share = None

            missing_value_message = "Non communiqué"

            #City Hall elections data
            elected_candidate_surname = cityhall_elec_data["Nom"]
            elected_candidate_name = cityhall_elec_data["Prénom"]
            elected_candidate_list = cityhall_elec_data["Liste"]
            elected_candidate_score = cityhall_elec_data["% Voix/Exp"]
            elected_candidate_color = cityhall_elec_data["bloc_text"]
            elected_candidate_def = cityhall_elec_data["Definition"]
            elected_candidate_party = cityhall_elec_data["LibNua"]
            elected_candidate_nucode = cityhall_elec_data["Code Nuance"]
            election_tour = cityhall_elec_data["Tour"]
            election_tour_string = "1er tour" if election_tour == 1 else "2ème tour"

            #Presidential elections data
            # pres_election_data_city = pres_elec_data[pres_elec_data["CODGEO"] == insee_city_id]
            pres_elec_data["Nom_prénom"] = pres_elec_data["Prénom"] + " " + pres_elec_data["Nom"]
            pres_elec_data_first_round = pres_elec_data.query("Tour == 1")
            pres_elec_data_second_round = pres_elec_data.query("Tour == 2")

            #Graph age population
            fig_age_pop = graph_housing(insee_age_pop, city)

            #Graph surface
            fig_housing_size = graph_housing(insee_surface, city)

            #Graph tour 1
            try:
                fig1, ax1 = plt.subplots(1, figsize=(6, 3))
                ax1 = sns.barplot(data = pres_elec_data_first_round,x='% Voix/Exp',y='Nom_prénom', palette="rocket", orient='h')
                ax1.set_ylabel("")
                ax1.set_xlabel("")
                sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=True, offset=None, trim=False)
                score_percent = round(pres_elec_data_first_round["% Voix/Exp"] * 100,1).astype(str) + "%"
                label = f"{score_percent} %"
                ax1.bar_label(ax1.containers[0], labels = score_percent,fmt='%.f')
                ax1.set_facecolor('white')
                ax1.set(xticklabels=[])
            except:
                fig1="Non communiqué"

            #Graph tour 2
            try:
                fig2, ax2 = plt.subplots(1, figsize=(6, 3))
                ax2 = sns.barplot(data = pres_elec_data_second_round,x='% Voix/Exp',y='Nom_prénom', palette="rocket", orient='h')
                ax2.set_ylabel("")
                ax2.set_xlabel("")
                sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=True, offset=None, trim=False)
                score_percent = round(pres_elec_data_second_round["% Voix/Exp"] * 100,1).astype(str) + "%"
                label = f"{score_percent} %"
                ax2.bar_label(ax2.containers[0], labels = score_percent,fmt='%.f')
                ax2.set_facecolor('white')
                ax2.set(xticklabels=[])
            except:
                fig2="Non communiqué"


            # REAL ESTATE DATA
            ## price per square meter
            dvf_preproc_df = dvf_preproc(dvf_2023)
            dvf_avg = dvf_per_city(dvf_preproc_df)

            try:
                apt_price_avg = dvf_avg.loc[dvf_avg["Type local"] == "Appartement","Valeur fonc / surface habitable"].values[0]
                apt_nb_transac = dvf_preproc_df["Type local"].value_counts()["Appartement"]
            except IndexError:
                apt_price_avg = None
            try:
                house_price_avg = dvf_avg.loc[dvf_avg["Type local"] == "Maison","Valeur fonc / surface habitable"].values[0]
                house_nb_transac = dvf_preproc_df["Type local"].value_counts()["Maison"]
            except IndexError:
                house_price_avg = None

            ## rental
            try:
                type_r = 'apt'
                rental_apt_city = rental.loc[
                    rental["type"] == type_r,
                    "loypredm2"].values[0]
                rental_nobs_apt_city = rental.loc[
                    rental["type"] == type_r,
                    "nbobs_com"].values[0]
            except IndexError:
                rental_apt_city = None
            try:
                type_r = 'house'
                rental_house_city = rental.loc[
                    rental["type"] == type_r,
                    "loypredm2"].values[0]
                rental_nobs_house_city = rental.loc[
                    rental["type"] == type_r,
                    "nbobs_com"].values[0]
            except IndexError:
                rental_house_city = None


            # WHEATHER DATA
            stations_coord = wheather_station_list(data_meteo_preproc_df)
            closest_station = find_closest_station(stations_coord, (city_lat, city_long))
            closest_station_name = stations_coord[stations_coord["numer_sta"] == closest_station]["nom_epci"].values[0]
            data_meteo_city = data_meteo_preproc_df.query("numer_sta == @closest_station").reset_index(drop=True)
            temp_city = temp_by_season(data_meteo_city)
            data_meteo_nat = data_meteo_national_avg(data_meteo_preproc_df)
            pluvio_avg_city = pluvio_moyenne(data_meteo_city)
            pluvio_avg_nat = pluvio_moyenne(data_meteo_nat)
            pluvio_df = pd.merge(pluvio_avg_city, pluvio_avg_nat, how="left", on=["month","month_name"], suffixes=('_city', '_nat'))\
                                .rename(columns={"precipitations_3h_city": city,"precipitations_3h_nat":"Moy. villes françaises"})
            pluvio_fig = graph_pluvio(pluvio_df, city)

            # POLLUTION DATA
            poll_station_coord = pollution_station_list(data_pollution_df)
            poll_closest_station = find_closest_station(poll_station_coord, (city_lat, city_long))
            poll_zas_name, poll_site_name = poll_station_coord[poll_station_coord["numer_sta"] == poll_closest_station]["Zas"].values[0], poll_station_coord[poll_station_coord["numer_sta"] == poll_closest_station]["nom site"].values[0]
            poll_data_city = data_pollution_df[data_pollution_df["code site"] == poll_closest_station].reset_index(drop=True)
            poll_avg_city = poll_data_city.groupby("date_clean", as_index=False)["valeur"].mean().reset_index(drop=True)
            poll_avg_nat = data_pollution_df.groupby("date_clean", as_index=False)["valeur"].mean().reset_index(drop=True)
            poll_df = pd.merge(poll_avg_city, poll_avg_nat, how="left", on=["date_clean"], suffixes=('_city', '_nat'))\
                                .rename(columns={"valeur_city": city,"valeur_nat":"Moy. nationale"})
            reco_OMS = 15
            poll_df["Recommandation OMS"] = reco_OMS
            poll_fig = graph_poll(poll_df, city, reco_OMS)

            # TAXES
            if (insee_city_id[:3] == '132') & (len(insee_city_id) == 5):
                taxes_city = taxes[taxes["Libellé commune"] == "MARSEILLE"]
            elif (insee_city_id[:3] == '693') & (len(insee_city_id) == 5):
                taxes_city = taxes[taxes["Libellé commune"] == "LYON"]
            elif (insee_city_id[:3] == '751') & (len(insee_city_id) == 5):
                taxes_city = taxes[taxes["Libellé commune"] == "VILLE DE PARIS"]
            else:
                taxes_city = taxes[taxes["code INSEE"] == insee_city_id]
            tfb_city = taxes_city["TFB"].values[0]
            teom_city = taxes_city["TEOM"].values[0]
            tfb_nat = taxes["TFB"].mean()
            teom_nat = taxes["TEOM"].mean()

            #################################### DISPLAYING DATA #######################################################

            st.map(pd.DataFrame({"lat":city_lat, "lon":city_long}, index=[0]))

            st.markdown("""___""")

            st.subheader("🌡️ Températures")
            st.caption("Moyenne 2020-2023")
            temp1, temp2, temp3, temp4=st.columns(4,gap='large')
            with temp1:
                    st.metric(label="☀️ Eté", value=f"{round(temp_city['été'], 1)} °C")
            with temp2:
                    st.metric(label="🍂 Automne", value=f"{round(temp_city['automne'], 1)} °C")
            with temp3:
                    st.metric(label="❄️ Hiver", value=f"{round(temp_city['hiver'], 1)} °C")
            with temp4:
                    st.metric(label="🌼 Printemps", value=f"{round(temp_city['printemps'], 1)} °C")

            st.write("##")
            st.subheader("🌧️ Pluviométrie")
            st.caption("Moyenne 2020-2023")
            st.write(pluvio_fig)
            st.caption(f"Station de mesure : **{closest_station_name}**")

            st.write("##")
            st.subheader("🏭 Pollution")
            st.caption(f"Niveau de particules fines (< 2.5 µg/m3) - Mesuré sur le site : {poll_zas_name} - {poll_site_name}")
            st.write(poll_fig)
            st.markdown("💡 Les particules fines, ou PM2.5, sont émises principalement lors des phénomènes de combustion et altèrent la santé respiratoire et cardiovasculaire")
            st.markdown("💡 L'OMS recommande de ne pas dépasser **5 µg/m3 de concentration moyenne par an** et **15 µg/m3 de concentration moyenne par jour** (i.e 3 à 4 jours de dépassement par an) ")
            st.caption("**Source :** 'WHO 2021 Air quality guidelines: Global update 2021'")

            st.markdown("""___""")

            st.subheader("📋 Recensement 2020")

            total1, total2, total3=st.columns(3,gap='large')
            with total1:
                    st.info('Population',icon="👫")
                    st.metric(label="Nb habitants",value=f"{pop_city:,.0f}".replace(","," "),delta=f"{pop_growth_city * 100:,.0f} % vs 2014")
            with total2:
                    st.info('Taux de pauvreté 2020',icon="💰")
                    metric_label = "Taux pauvreté"
                    if pd.isna(poverty_rate):
                        st.metric(label=metric_label,value=missing_value_message)
                    else:
                        st.metric(label=metric_label,value=f"{poverty_rate:,.0f} %")
                    st.text(f'France entière: {poverty_rate_median:,.0f} %')
            with total3:
                    st.info('Taux de chômage',icon="👔")
                    metric_label = "Nb chômeurs / Population active"
                    if pd.isna(unemployment_rate_city):
                        st.metric(label=metric_label,value=missing_value_message)
                    else:
                        st.metric(label=metric_label,value=f"{unemployment_rate_city * 100:,.0f} %")
                    st.text(f'France entière: {unemployment_rate_median * 100:,.0f} %')

            total4, total5, total6=st.columns(3,gap='large')
            with total4:
                    st.info('Part de HLM',icon="🏢")
                    metric_label = "Nb HLM / Nb total logements"
                    if pd.isna(hlm_share):
                        st.metric(label=metric_label,value=missing_value_message)
                    else:
                        st.metric(label=metric_label,value=f"{hlm_share * 100:,.0f} %")
                        st.text(f'France entière : {hlm_share_nat * 100:,.0f} %')
            with total5:
                    st.info('Part de propriétaires',icon="🔑")
                    metric_label = "Nb propriétaires / Nb total logements"
                    if pd.isna(owner_share):
                        st.metric(label=metric_label,value=missing_value_message)
                    else:
                        st.metric(label=metric_label,value=f"{owner_share * 100:,.0f} %")
                        st.text(f'France entière : {owner_share_nat * 100:,.0f} %')
            with total6:
                    st.info("Part d'immigrés",icon="🌏")
                    metric_label = "Nb immigrés / Population"
                    if pd.isna(immigration_share):
                        st.metric(label=metric_label,value=missing_value_message)
                    else:
                        st.metric(label="Nb immigrés / Nb population",value=f"{immigration_share * 100:,.0f} %")
                        st.text(f'France entière : {immigration_share_nat * 100:,.0f} %')

            total7, total8=st.columns(2,gap='large')
            with total7:
                st.info('Age du référent du ménage',icon="👴🏼")
                st.write(fig_age_pop)
            with total8:
                st.info('Taille des logements',icon="📐")
                st.write(fig_housing_size)

            st.write("##")
            st.caption("**Source :** INSEE | 2020")

            st.markdown("""___""")

            st.subheader("Education")

            binary_col = [
                "ecole_maternelle",
                "ecole_elementaire",
                "voie_generale",
                "voie_technologique",
                "voie_professionnelle",
                "section_internationale",
                "section_europeenne",
                "segpa",
                "restauration",
                "hebergement"]
            for col in binary_col:
                 schools_data[col] = replace_checkmark(schools_data[col])

            tab1, tab2, tab3, tab4 = st.tabs(["Ecoles", "Collèges", "Lycées", "Autres"])
            with tab1:
                 data_ec = schools_data.query("type_etablissement == 'Ecole'").drop(columns=[
                    "CODGEO",
                    "identifiant_de_l_etablissement",
                    "voie_generale",
                    "voie_technologique",
                    "voie_professionnelle",
                    "section_internationale",
                    "section_europeenne",
                    "segpa",
                    "hebergement",
                    "libelle_nature",
                    "taux_brut_de_reussite_total_series",
                    "taux_mention_brut_toutes_series"
                 ])
                 st.dataframe(data_ec, hide_index=True)
            with tab2:
                 data_col = schools_data.query("type_etablissement == 'Collège'").drop(columns=[
                    "CODGEO",
                    "identifiant_de_l_etablissement",
                    "ecole_maternelle",
                    "ecole_elementaire",
                    "voie_generale",
                    "voie_technologique",
                    "voie_professionnelle",
                    "libelle_nature",
                    "taux_brut_de_reussite_total_series",
                    "taux_mention_brut_toutes_series"
                 ])
                 st.dataframe(data_col, hide_index=True)
            with tab3:
                 data_lyc = schools_data.query("type_etablissement == 'Lycée'").drop(columns=[
                    "CODGEO",
                    "identifiant_de_l_etablissement",
                    "ecole_maternelle",
                    "ecole_elementaire"
                 ])
                 st.dataframe(data_lyc, hide_index=True)
            with tab4:
                 data_autre = schools_data.query("type_etablissement == 'Autre'").drop(columns=[
                    "CODGEO",
                    "identifiant_de_l_etablissement"
                 ])
                 st.dataframe(data_autre, hide_index=True)

            st.write("##")
            st.caption("**Source :** Ministère de l'Education Nationale et de la Jeunesse | 2022")

            st.markdown("""---""")
            st.subheader("Résultat aux élections municipales de 2020")

            dict_color_pol = {"Extrême Gauche": '🔴',
                "Gauche":'🟠',
                "Divers":'🟢',
                "Centre": '⚪️',
                "Droite":'🔵',
                "Extrême Droite": '⚫️'}
            pol3, pol4 = st.columns((1,8), gap='small')
            with pol3:
                image = Image.open('processed_data/interpro_Maire.png')
                st.image(image, width=140)
            with pol4:
                st.markdown(f"**{elected_candidate_surname}** **{elected_candidate_name}**")
                st.caption(f"Elu(e) au {election_tour_string} | **Score :** {elected_candidate_score * 100:,.1f} % | **Liste :** {elected_candidate_list}")
                st.caption(f"**Parti ou mouvement associé :** {elected_candidate_party}" if elected_candidate_nucode != 'NC' else 'Sans étiquette')
                st.caption(f"**Bloc :** {elected_candidate_color} {dict_color_pol[elected_candidate_color]}" if elected_candidate_color is not None else '')


            st.write("##")
            st.subheader("Résultat aux élections présidentielles de 2022")

            item5, item6 = st.columns(2,gap='large')
            with item5:
                st.markdown("#### Premier tour")
                st.write(fig1)
            with item6:
                st.markdown("#### Second tour")
                st.write(fig2)

            st.markdown("""___""")

            st.subheader("Prix immobilier")
            st.markdown("*Moyenne des transactions du premier semestre 2023*")

            item7, item8 = st.columns(2,gap='large')
            with item7:
                st.info("Appartements",icon="🏬")
                st.metric(label="Valeur foncière / Surface habitable",value=f"{apt_price_avg:,.0f} €/m²".replace(","," ") if apt_price_avg else "Aucune transaction")
                if apt_price_avg:
                    st.text(f"Nb transactions : {apt_nb_transac}")
            with item8:
                st.info("Maisons",icon="🏚️")
                st.metric(label="Valeur foncière / Surface habitable",value=f"{house_price_avg:,.0f} €/m²".replace(","," ") if house_price_avg else "Aucune transaction")
                if house_price_avg:
                    st.text(f"Nb transactions : {house_nb_transac}")

            st.write("##")

            st.markdown("**Détail des transactions du premier semestre 2023**")
            dvf_2023_clean = dvf_preproc_df.drop(columns=["CODGEO"]).reset_index(drop=True)
            st.dataframe(dvf_2023_clean, hide_index=True, width=1500)

            st.write("##")

            st.subheader("Loyer moyen")
            st.markdown("*Moyenne du T3 2022*")

            rent1, rent2 = st.columns(2,gap='large')
            with rent1:
                st.info("Appartements",icon="🏬")
                st.metric(label="Loyer d'annonce charges comprises pour un non meublé",value=f"{rental_apt_city:,.0f} €/m²" if rental_apt_city else "Aucune annonce")
                if rental_apt_city:
                    st.text(f"Nb annonces : {rental_nobs_apt_city}")
            with rent2:
                st.info("Maisons",icon="🏚️")
                st.metric(label="Loyer d'annonce charges comprises pour un non meublé",value=f"{rental_house_city:,.0f} €/m²" if rental_house_city else "Aucune annonce")
                if rental_house_city:
                    st.text(f"Nb annonces : {rental_nobs_house_city}")

            st.markdown("💡 *Les indicateurs de loyer sont calculés par l'Agence Nationale pour l'Information sur le Logement (ANIL), grâce à l'utilisation des données d'annonces parues sur Leboncoin et SeLoger au T3 2022*")

            st.write("##")

            st.subheader("Impôts locaux")
            st.markdown("*Votés en 2022*")

            item9, item10 = st.columns(2,gap='large')
            with item9:
                st.info("Taxe foncière",icon="💶")
                st.metric(label="Taxe foncière sur les propriétés bâties (TFPB)",value=f"{tfb_city:,.0f} %")
                st.text(f'Moyenne nationale : {tfb_nat:,.0f} %')
            with item10:
                st.info("Taxe d'enlèvement des ordures ménagères",icon="🗑️")
                st.metric(label="TEOM - Taux plein net",value=f"{teom_city:,.0f} %")
                st.text(f'Moyenne nationale : {teom_nat:,.0f} %')
