import pandas as pd
import numpy as np
import calendar
import geopy.distance
import streamlit as st

@st.cache_data
def load_data(file_path: str):
    return pd.read_parquet(file_path)

def dvf_preproc(dvf_clean_df):
    dvf_house_apt = dvf_clean_df.loc[
        (dvf_clean_df["Type local"].isin(["Maison", "Appartement"])) &
        (dvf_clean_df["Nature mutation"] == "Vente")
        ].reset_index(drop=True)

    dvf_preproc_unique = dvf_house_apt.drop_duplicates(subset=["Date mutation","Valeur fonciere", "No voie", "Voie"])
    lower_threshold = dvf_preproc_unique["Valeur fonc / surface habitable"].quantile(.05)
    upper_threshold = dvf_preproc_unique["Valeur fonc / surface habitable"].quantile(.95)
    dvf_preproc_clean = dvf_preproc_unique.copy()
    dvf_preproc_clean["Valeur fonc / surface habitable"] = dvf_preproc_clean["Valeur fonc / surface habitable"].clip(lower=lower_threshold, upper=upper_threshold)
    dvf_preproc_no_outlier = dvf_preproc_clean[~dvf_preproc_clean["Valeur fonc / surface habitable"].isin([lower_threshold, upper_threshold])]

    return dvf_preproc_no_outlier

def dvf_per_city(dvf_preproc_df):
    dvf_house_apt_avg = dvf_preproc_df.groupby(['CODGEO','Type local'], as_index = False)["Valeur fonc / surface habitable"].mean().sort_values(by=["CODGEO","Type local"]).reset_index(drop=True)
    return dvf_house_apt_avg

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
    df["month_name"] = df["date_clean"].dt.month.apply(lambda x : calendar.month_name[x])
    df["year_month"] = df["date_clean"].dt.to_period("M")
    df_sum_month = df.groupby(["year_month", "month_name"], as_index=False)["precipitations_3h"].sum()
    df_sum_month["month"] = df_sum_month["year_month"].dt.month
    df_avg_month = df_sum_month.groupby(["month_name","month"], as_index=False)["precipitations_3h"].mean().sort_values(by=["month"]).reset_index(drop=True)
    return df_avg_month

def replace_checkmark(data: pd.Series):
     data_checkmark = data.replace(1,"✔️").replace(0,"❌")
     return data_checkmark
