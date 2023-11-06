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
    ax = sns.barplot(data_format, x="value", y ="LIB_MOD", hue="variable", palette="mako", orient='h')
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    score_percent_city = round(data_format.query("variable == @city")["value"] * 100,1).astype(str) + "%"
    score_percent_nat = round(data_format.query("variable == 'France entière'")["value"] * 100,1).astype(str) + "%"
    ax.bar_label(ax.containers[0], labels = score_percent_city,fmt='%.f')
    ax.bar_label(ax.containers[1], labels = score_percent_nat,fmt='%.f')

    ax.set_ylabel(None)
    ax.set_xlabel(None)
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
    
    dvf_preproc_unique = dvf_house_apt.drop_duplicates(subset=["Date mutation","Valeur fonciere", "No voie", "Voie"])
    lower_threshold = dvf_preproc_unique["Valeur fonc / surface habitable"].quantile(.1)
    upper_threshold = dvf_preproc_unique["Valeur fonc / surface habitable"].quantile(.9)
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
    df["month_name"] = df["date_clean"].dt.month_name(locale = 'French')
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
    ax = sns.barplot(data_format, x="month_name", y ="value", hue="variable", palette="mako")
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax.legend().set_title(None)
    ax.set_ylabel("mm")
    ax.set_xlabel(None)
    sns.set_style("darkgrid")
    return fig

#################################### WELCOME PAGE #######################################################

st.set_page_config(
    page_title="City_Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

folder = "processed_data"

title_image, title_text, title_image_next = st.columns((1,4,1))
with title_image:
    image_france = Image.open('processed_data/logo-france-2.png')
    st.image(image_france, width=120)
with title_text:
    st.markdown('# *Les villes :blue[Françaises] en chiffres*')
with title_image_next:
    image_france = Image.open('processed_data/logo-france-2.png')
    st.image(image_france, width=120)
     

folder = "processed_data"
insee_data_file = rf"{folder}\2020_insee_data_population.parquet.gzip"
insee_data_sum_stat = rf"{folder}\2020_insee_data_population_sum_stat.parquet.gzip"
cityhall_elec_data_file = rf"{folder}\2020_gouv_data_city_hall_elections.parquet.gzip"
pres_elec_data_file = rf"{folder}\2022_gouv_data_pres_elections.parquet.gzip"
insee_city_name_id = rf"{folder}\cities_insee_id.parquet.gzip"
insee_housing_const = rf"{folder}\2020_housing_constr_period.parquet.gzip"
insee_hlm = rf"{folder}\2020_housing_hlm.parquet.gzip"
insee_immigration = rf"{folder}\2020_housing_immigration.parquet.gzip"
insee_nb_rooms = rf"{folder}\2020_housing_nb_rooms.parquet.gzip"
insee_owner_share = rf"{folder}\2020_housing_owner_share.parquet.gzip"
insee_housing_size = rf"{folder}\2020_housing_size.parquet.gzip"
insee_housing_age = rf"{folder}\2020_housing_age.parquet.gzip"
real_estate_2023 = rf"{folder}\2023_real_estate_mkt.parquet.gzip"
wheather_data_2020_2023 = rf"{folder}\2020_2023_wheather.parquet.gzip"

insee_city_name_df = load_data(insee_city_name_id)

list_cities = list(insee_city_name_df["LIBELLE_COM"].unique())

# Selecting the city
city = st.selectbox(
    'Sélectionner la ville :',
    list_cities,
    index=3
    )

launch_button = st.button("Afficher les données", type="primary")

if launch_button:
    
    #################################### LOADING DATA #######################################################

    #Geographic data on selected city
    insee_city_id = insee_city_name_df[insee_city_name_df["LIBELLE_COM"] == city]["COM"].iloc[0].lstrip("0")
    insee_city_dep_name = insee_city_name_df[insee_city_name_df["LIBELLE_COM"]== city]["LIBELLE_DEP"].iloc[0]
    insee_city_dep_num = insee_city_name_df[insee_city_name_df["LIBELLE_COM"]== city]["DEP"].iloc[0]
    insee_city_reg = insee_city_name_df[insee_city_name_df["LIBELLE_COM"]== city]["LIBELLE_REG"].iloc[0]

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
    insee_nb_rooms_data = pd.read_parquet(insee_nb_rooms, filters = criteria)
    insee_constr_period = pd.read_parquet(insee_housing_const, filters = criteria)
    insee_age_pop = pd.read_parquet(insee_housing_age, filters = criteria)
    insee_surface = pd.read_parquet(insee_housing_size, filters = criteria)
    dvf_2023 = pd.read_parquet(real_estate_2023, filters = criteria)


    #################################### STORING VALUES #######################################################

    #Importing data on wheather station
    data_meteo_preproc_df = pd.read_parquet(wheather_data_2020_2023)

    #Insee data - Getting the national values for comparison
    pop_median = insee_sum_stats.loc["Median","P20_POP"]
    poverty_rate_median = insee_sum_stats.loc["Median","TP6020"]
    standard_of_living_median = insee_sum_stats.loc["Median","MED20"]
    principal_residency_median = insee_sum_stats.loc["Median","RP_SHARE"]
    unemployment_rate_median = insee_sum_stats.loc["Median","CHOM_RATE"]
    empty_residency_median = insee_sum_stats.loc["Median","LOGVAC_RATE"]
    hlm_share_nat = insee_hlm_data["proportion_national"].values[0]
    immigration_share_nat = insee_immigration_data.query("IMMIM == '1'")["proportion_national"].values[0]
    owner_share_nat = insee_owner_share_data["proportion_national"].values[0]

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
    hlm_share = insee_hlm_data["proportion_city"].values[0]
    immigration_share = insee_immigration_data.query("IMMIM == '1'")["proportion_city"].values[0]
    owner_share = insee_owner_share_data["proportion_city"].values[0]

    missing_value_message = "Non communiqué"

    #City Hall elections data
    # city_election_data = cityhall_elec_data[cityhall_elec_data["CODGEO"] == insee_city_id].iloc[0]
    elected_candidate_surname = cityhall_elec_data["Nom"]
    elected_candidate_name = cityhall_elec_data["Prénom"]
    elected_candidate_list = cityhall_elec_data["Liste"]
    elected_candidate_score = cityhall_elec_data["% Voix/Exp"]
    election_tour = cityhall_elec_data["Tour"]
    election_tour_string = "1er tour" if election_tour == 1 else "2ème tour"

    #Presidential elections data
    # pres_election_data_city = pres_elec_data[pres_elec_data["CODGEO"] == insee_city_id]
    pres_elec_data["Nom_prénom"] = pres_elec_data["Prénom"] + " " + pres_elec_data["Nom"]
    pres_elec_data_first_round = pres_elec_data.query("Tour == 1")
    pres_elec_data_second_round = pres_elec_data.query("Tour == 2")

    #Graph age population
    fig_age_pop = graph_housing(insee_age_pop, city)

    #Graph nb rooms
    fig_nb_rooms = graph_housing(insee_nb_rooms_data, city)

    #Graph constr_period
    fig_constr_period = graph_housing(insee_constr_period, city)

    #Graph surface
    fig_housing_size = graph_housing(insee_surface, city)

    #Graph tour 1
    fig1, ax1 = plt.subplots(1, figsize=(6, 3))
    ax1 = sns.barplot(pres_elec_data_first_round,x='% Voix/Exp',y='Nom_prénom', palette="rocket", orient='h')
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    score_percent = round(pres_elec_data_first_round["% Voix/Exp"] * 100,1).astype(str) + "%"
    label = f"{score_percent} %"
    ax1.bar_label(ax1.containers[0], labels = score_percent,fmt='%.f')

    #Graph tour 2
    fig2, ax2 = plt.subplots(1, figsize=(6, 3))
    ax2 = sns.barplot(pres_elec_data_second_round,x='% Voix/Exp',y='Nom_prénom', palette="rocket", orient='h')
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=False, offset=None, trim=False)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    score_percent = round(pres_elec_data_second_round["% Voix/Exp"] * 100,1).astype(str) + "%"
    label = f"{score_percent} %"
    ax2.bar_label(ax2.containers[0], labels = score_percent,fmt='%.f')


    # REAL ESTATE DATA
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

    # WHEATHER DATA
    stations_coord = wheather_station_list(data_meteo_preproc_df)
    closest_station = find_closest_station(stations_coord, (city_lat, city_long))
    data_meteo_city = data_meteo_preproc_df.query("numer_sta == @closest_station").reset_index(drop=True)
    temp_city = temp_by_season(data_meteo_city)
    data_meteo_nat = data_meteo_national_avg(data_meteo_preproc_df)
    pluvio_avg_city = pluvio_moyenne(data_meteo_city)
    pluvio_avg_nat = pluvio_moyenne(data_meteo_nat)
    pluvio_df = pd.merge(pluvio_avg_city, pluvio_avg_nat, how="left", on=["month","month_name"], suffixes=('_city', '_nat'))\
                        .rename(columns={"precipitations_3h_city": city,"precipitations_3h_nat":"Moy. villes françaises"})
    pluvio_fig = graph_pluvio(pluvio_df, city)

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

    st.markdown("""___""")
    
    st.markdown("*Source:* **INSEE / 2020**")
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
            st.metric(label="Nb HLM / Nb total logements",value=f"{hlm_share * 100:,.0f} %")
            st.text(f'France entière : {hlm_share_nat * 100:,.0f} %')
    with total5:
            st.info('Part de propriétaires',icon="🔑")
            st.metric(label="Nb propriétaires / Nb total logements",value=f"{owner_share * 100:,.0f} %")
            st.text(f'France entière : {owner_share_nat * 100:,.0f} %')
    with total6:
            st.info("Part d'immigrés",icon="🌏")
            st.metric(label="Nb immigrés / Nb population",value=f"{immigration_share * 100:,.0f} %")
            st.text(f'France entière : {immigration_share_nat * 100:,.0f} %')
    
    total7, total8=st.columns(2,gap='large')
    with total7:
        st.info('Age du référent du ménage',icon="👴🏼")
        st.write(fig_age_pop)
    with total8:
        st.info('Années de construction des logements',icon="🏗️")
        st.write(fig_constr_period)
        
    total9, total10=st.columns(2,gap='large')
    with total9:
        st.info('Nb de pièces par appartement',icon="🛏️")
        st.write(fig_nb_rooms)
    with total10:
        st.info('Taille des logements',icon="📐")
        st.write(fig_housing_size)

    st.markdown("""---""")
    st.subheader("Résultat aux élections municipales de 2020")
    
    item3, item4 = st.columns((1,4))
    with item3:
        image = Image.open('processed_data/interpro_Maire.png')
        st.image(image, width=140)
    with item4:
        st.markdown(f"**{elected_candidate_surname}** **{elected_candidate_name}**")
        st.caption(f"Elu(e) au {election_tour_string}")
        st.caption(f"**Score :** {elected_candidate_score * 100:,.1f} %")
        st.caption(f"**Liste :** {elected_candidate_list}")

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

    # on = st.button('Afficher le détail des transactions')
    # if (launch_button) & (on) :

    st.write("##")
    st.markdown("**Détail des transactions du premier semestre 2023**")
    dvf_2023_clean = dvf_preproc_df.drop(columns=["CODGEO"]).reset_index(drop=True)
    st.dataframe(dvf_2023_clean, hide_index=True, width=1500)