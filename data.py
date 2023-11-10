import requests, zipfile, io, json
import pandas as pd
import numpy as np
import os

def download_data_insee(zip_file_url: str, csv_file_name: str) -> pd.DataFrame:

    """
    Download the data on French population per city. Source : Insee

    Returns:
        A dataframe of the csv
    """

    r = requests.get(zip_file_url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    col_to_use = ["CODGEO","TP6020","MED20","P20_POP","P14_POP","P20_RP","P20_LOG",
              "P20_CHOM1564","P20_ACT1564","P20_LOGVAC"]
    with z as zipf:
        csv_file = zipf.open(csv_file_name)
        df = pd.read_csv(csv_file, sep=";", low_memory=False, usecols = col_to_use)

    return df


def download_data_housing_insee(url_parquet: str) -> pd.DataFrame:

    """
    Download the data on French housing in 2020. Source : Insee

    Returns:
        A dataframe of the parquet
    """

    col_to_use = ["COMMUNE","ARM","ACHL","HLML","IMMIM","INPER","NBPI","STOCD",
              "SURF","AGEMEN8"]
    
    df = pd.read_parquet(url_parquet, columns = col_to_use)

    return df


def read_data_from_local_drive(zip_file_name: str, csv_file_name: str) -> pd.DataFrame:

    """
    Read the data on French population per city, previously saved as .zip in the same folder as this script. Source : Insee

    Returns:
        A dataframe of the csv
    """

    z = zipfile.ZipFile(zip_file_name)
    col_to_use = ["CODGEO","TP6020","MED20","P20_POP","P14_POP","P20_RP","P20_LOG",
              "P20_CHOM1564","P20_ACT1564","P20_LOGVAC"]
    with z as zipf:
        csv_file = zipf.open(csv_file_name)
        df = pd.read_csv(csv_file, sep=";", header=0, low_memory=False, usecols = col_to_use)

    return df

def read_meta_data_from_local_drive(zip_file_name: str, csv_file_name: str) -> pd.DataFrame:

    """
    Read the meta data on French population per city, previously saved as .zip in the same folder as this script. Source : Insee

    Returns:
        A dataframe of the csv
    """

    z = zipfile.ZipFile(zip_file_name)
    with z as zipf:
        csv_file = zipf.open(csv_file_name)
        df = pd.read_csv(csv_file, sep=";", header=0, low_memory=False)

    return df

def load_insee_data(from_url:bool)->pd.DataFrame:

    data_pop_csv = "base_cc_comparateur.csv"
    meta_data_pop_csv = "meta_base_cc_comparateur.csv"
    
    if from_url:
        zip_file_url = 'https://www.insee.fr/fr/statistiques/fichier/2521169/base_cc_comparateur_csv.zip'
        data_pop_per_city = download_data_insee(zip_file_url, data_pop_csv)
        meta_data_pop_per_city = download_data_insee(zip_file_url, meta_data_pop_csv)
    else:
        zip_name = "sources/base_cc_comparateur_csv.zip"
        data_pop_per_city = read_data_from_local_drive(zip_name,data_pop_csv)
        meta_data_pop_per_city = read_meta_data_from_local_drive(zip_name,meta_data_pop_csv)
    
    return (data_pop_per_city, meta_data_pop_per_city)

def load_dvf(url):

    col_to_keep = [
        "Date mutation",
        "Nature mutation",
        "Valeur fonciere",
        "Commune",
        'Code departement',
        'Code commune',
        'No voie',
        'Type de voie',
        'Voie',
        'Type local',
        'Surface reelle bati', 
        'Nombre pieces principales',
        'Surface terrain'
    ]
    dvf = pd.read_csv(url, sep="|", usecols=col_to_keep, low_memory=False)
    return dvf

def load_cities_geo_info():
    with open('sources/cities.json', 'r') as f:
    # Read the JSON data from the file
        data = json.load(f)
    return data

def load_city_election_first_round(file_first_round: str):
    first_round_election = pd.read_excel(file_first_round)
    return first_round_election

def load_city_election_second_round(file_second_round: str):
    second_round_election_init = pd.read_csv(file_second_round, header=0, sep=';',encoding='latin_1',low_memory=False)
    return second_round_election_init

def load_presidential_election(file_pres_election: str):
    df = pd.read_excel(file_pres_election, header=0)
    return df

def load_meteo_data(path_json):

    data_meteo_df = pd.read_json(path_json)

    return data_meteo_df

def formatting_metropoles_votes(second_round_election_init):
    df = second_round_election_init.copy()
    df["Code Officiel Commune_Modif"] = np.where(
        df["Nom Officiel Commune"].isin(["Paris","Lyon","Marseille"]),
        df["Code Officiel Commune"] + "SR" + df["Code B.Vote"].str[:-2].str.rjust(2,'0'),
        df["Code Officiel Commune"])
    df["Code Officiel Commune"] = df["Code Officiel Commune_Modif"]
    df.drop(columns= ["Code Officiel Commune_Modif"], inplace=True)
    
    return df

def votes_per_city(second_round_election_init):
    cols_to_group = ["Code Officiel Commune", "Code Officiel du département", 
            "Nom Officiel Commune","Nom","Prénom","Code Nuance","Liste"]
    cols_to_sum = ["Voix","Exprimés"]
    second_round_election = second_round_election_init.groupby(cols_to_group, as_index=False)[cols_to_sum].sum()
    second_round_election["% Voix/Exp"] = second_round_election["Voix"] / second_round_election["Exprimés"]
    return second_round_election

def presidential_votes_per_city(presidential_election_init):
    cols_to_group = ["CODGEO"]
    cols_to_sum = ["Voix","Exprimés"]
    second_round_election = second_round_election_init.groupby(cols_to_group, as_index=False)[cols_to_sum].sum()
    second_round_election["% Voix/Exp"] = second_round_election["Voix"] / second_round_election["Exprimés"]
    return second_round_election

def election_first_round_formatting(first_round_election: pd.DataFrame, n_min_candidates: int, n_max_candidates: int):
    code_dep_list = []
    code_com_list = []
    lib_com_list = []
    surname_list = []
    name_list = []
    code_nuance_list = []
    liste_list = []
    # votes_over_registered_list = []
    votes_over_expressed_list = []

    for index, row in first_round_election.iterrows():
        code_dep_list.append(row["Code du département"])
        code_com_list.append(row["Code de la commune"])
        lib_com_list.append(row["Libellé de la commune"])
        surname_list.append(row["Nom"])
        name_list.append(row["Prénom"])
        code_nuance_list.append(row["Code Nuance"])
        liste_list.append(row["Liste"])
        # votes_over_registered_list.append(row["% Voix/Ins"])
        votes_over_expressed_list.append(row["% Voix/Exp"])

        for i in range(n_min_candidates,n_max_candidates):
            code_dep_list.append(row["Code du département"])
            code_com_list.append(row["Code de la commune"])
            lib_com_list.append(row["Libellé de la commune"])
            surname_list.append(row["Nom"+"_"+str(i)])
            name_list.append(row["Prénom"+"_"+str(i)])
            code_nuance_list.append(row["Code Nuance"+"_"+str(i)])
            liste_list.append(row["Liste"+"_"+str(i)])
            # votes_over_registered_list.append(row["% Voix/Ins"+"_"+str(i)])
            votes_over_expressed_list.append(row["% Voix/Exp"+"_"+str(i)])

    result_election_t1 = pd.DataFrame(
        { 
            "Code du département": code_dep_list,
            "Code de la commune" : code_com_list,
            "Libellé de la commune" : lib_com_list,
            "Nom" : surname_list,
            "Prénom" : name_list,
            "Code Nuance": code_nuance_list,
            "Liste": liste_list,
            # "% Voix/Ins": votes_over_registered_list,
            "% Voix/Exp" : votes_over_expressed_list,
            "Tour": 1
        }
    )

    result_election_t1["% Voix/Exp"] = result_election_t1["% Voix/Exp"] / 100 
    result_election_t1["Code de la commune"] = result_election_t1["Code de la commune"].astype(str)
    result_election_t1["Code du département"] = result_election_t1["Code du département"].astype(str) 
    result_election_t1.insert(0,"Code Officiel Commune", result_election_t1["Code du département"] + result_election_t1["Code de la commune"].str.rjust(3,'0'))

    return result_election_t1

def presidential_election_formatting(pres_election: pd.DataFrame, n_min_candidates: int, n_max_candidates: int):
    code_dep_list = []
    code_com_list = []
    lib_com_list = []
    surname_list = []
    name_list = []
    votes_expr_list = []
    votes_candidate_list = []
    cod_geo_list = []

    for index, row in pres_election.iterrows():
        code_dep_list.append(row["Code du département"])
        code_com_list.append(row["Code de la commune"])
        lib_com_list.append(row["Libellé de la commune"])
        surname_list.append(row["Nom"])
        name_list.append(row["Prénom"])
        votes_expr_list.append(row["Exprimés"])
        votes_candidate_list.append(row["Voix"])
        cod_geo_list.append(row["CODGEO"])

        for i in range(n_min_candidates,n_max_candidates):
            code_dep_list.append(row["Code du département"])
            code_com_list.append(row["Code de la commune"])
            lib_com_list.append(row["Libellé de la commune"])
            surname_list.append(row["Nom"+"_"+str(i)])
            name_list.append(row["Prénom"+"_"+str(i)])
            votes_expr_list.append(row["Exprimés"])
            votes_candidate_list.append(row["Voix"+"_"+str(i)])
            cod_geo_list.append(row["CODGEO"])

    result_election = pd.DataFrame(
        { 
            "Code du département": code_dep_list,
            "Code de la commune" : code_com_list,
            "Libellé de la commune" : lib_com_list,
            "Nom" : surname_list,
            "Prénom" : name_list,
            "Exprimés": votes_expr_list,
            "Voix" : votes_candidate_list,
            "CODGEO": cod_geo_list
        }
    )

    result_election["Code de la commune"] = result_election["Code de la commune"].astype(str)
    result_election["Code du département"] = result_election["Code du département"].astype(str) 

    #Grouping by city and candidates
    cols_to_group = ["CODGEO","Code du département","Code de la commune","Libellé de la commune", "Nom", "Prénom"]
    cols_to_sum = ["Exprimés", "Voix"]
    result_election_per_city = result_election.groupby(by=cols_to_group, as_index=False)[cols_to_sum].sum().reset_index(drop=True)
    result_election_per_city["% Voix/Exp"] = result_election_per_city["Voix"] / result_election_per_city["Exprimés"] 

    return result_election_per_city


def election_second_round_formatting(second_round_election: pd.DataFrame):

    result_election_t2 = pd.DataFrame(
        { 
            "Code Officiel Commune": second_round_election["Code Officiel Commune"],
            "Code du département": second_round_election["Code Officiel du département"],
            "Code de la commune" : second_round_election["Code Officiel Commune"].str[-3:].str.lstrip("0"),
            "Libellé de la commune" : second_round_election["Nom Officiel Commune"],
            "Nom" : second_round_election["Nom"],
            "Prénom" : second_round_election["Prénom"],
            "Code Nuance": second_round_election["Code Nuance"],
            "Liste": second_round_election["Liste"],
            "% Voix/Exp" : second_round_election["% Voix/Exp"],
            "Tour": 2
        }
    )


    return result_election_t2

def concat_first_second_round(result_election_t1, result_election_t2):
    result_election = pd.concat([result_election_t1, result_election_t2]).sort_values(by = ["Libellé de la commune", "% Voix/Exp"], ascending=[True, False]).reset_index(drop=True)
    return result_election

def election_per_city_dataset_final(result_election):
    tour_city = result_election.groupby(["Code Officiel Commune"], as_index=False)["Tour"].max()
    winning_tour_election = pd.merge(
        tour_city, 
        result_election, 
        how="left", 
        on=["Code Officiel Commune","Tour"]
        ).\
        sort_values(by = ["Libellé de la commune", "% Voix/Exp"], ascending=[True, False]).\
        dropna(subset="Nom").\
        reset_index(drop=True)
    float_col = ["% Voix/Exp"]
    str_col = list(set(list(winning_tour_election.columns)) - set(float_col))
    winning_tour_election[str_col] = winning_tour_election[str_col].astype(str)
    winning_tour_election[float_col] = winning_tour_election[float_col].astype(float)
    
    return winning_tour_election

def load_election_city_2020():
    data_maires = requests.get("https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/election-france-municipale-2020-deuxieme-tour@public/records?select=com_name%2C%20nuance%2C%20surname%2C%20name%2C%20com_code%2C%20vote_exp")
    data_maires_json = data_maires.json()
    data_maires_json["results"]
    return data_maires_json

def load_election_city_2020_df(election_city_json):
    data_maires = election_city_json["results"]
    com_name = [candidate["com_name"] for candidate in data_maires]
    nuance = [candidate["nuance"] for candidate in data_maires]
    surname = [candidate["surname"] for candidate in data_maires]
    name = [candidate["name"] for candidate in data_maires]
    com_code = [candidate["com_code"] for candidate in data_maires]
    vote_exp = [candidate["vote_exp"] for candidate in data_maires]
    data_maires_df = pd.DataFrame(
        {
            "com_name":com_name, 
            "nuance":nuance, 
            "surname":surname, 
            "name":name,
            "com_code":com_code,
            "vote_exp":vote_exp
            }
            )
    return data_maires_df

def load_cities_geo_info_df(cities_geo_json):
    data_cities = cities_geo_json["cities"]
    insee_codes = [city["insee_code"] for city in data_cities]
    city_codes = [city["city_code"] for city in data_cities]
    lat = [city["latitude"] for city in data_cities]
    lon = [city["longitude"] for city in data_cities]
    data_cities_df = pd.DataFrame({"insee_code":insee_codes, "city_code":city_codes, "latitude":lat, "longitude":lon})
    return data_cities_df

def replace_value_metropole(df: pd.DataFrame, city_missing: str, city_to_use: str):
    #Replace the 'city_code', 'latitude' and 'longitude' in df for the city_missing by the city_to_use
    col_to_replace = ['city_code', 'latitude', 'longitude']
    for col in col_to_replace:
        df.loc[df['LIB_MOD'] == city_missing, col] = df.loc[df['LIB_MOD'] == city_to_use, col].values[0]
    return df

def cod_geo_result_election(result_election):
    df = result_election.copy()
    cod_geo = np.where(
        df["Code Officiel Commune"].str[:5] == "69123",
        "6938" + df["Code Officiel Commune"].str[-1:],
        np.where(
            df["Code Officiel Commune"].str[:5] == "13055",
            "132" + df["Code Officiel Commune"].str[-2:],
            np.where(
            df["Code Officiel Commune"].str[:5] == "75056",
            "751" + df["Code Officiel Commune"].str[-2:],
            df["Code Officiel Commune"]
    )))
    df["CODGEO"] = cod_geo
    
    return df

def cod_geo_result_presidential_election(result_presidential_election):

    df = result_presidential_election.copy()
    df["Code de la commune"] = df["Code de la commune"].astype(str)
    df["Code du b.vote"] = df["Code du b.vote"].astype(str)
    df["Code du département"] = df["Code du département"].astype(str)

    cod_geo = np.where(
        df["Libellé de la commune"] == "Lyon",
        "6938" + df["Code du b.vote"].str[:2].str.lstrip("0"),
        np.where(
            df["Libellé de la commune"] == "Marseille",
            "132" + df["Code du b.vote"].str[:2],
            np.where(
            df["Libellé de la commune"] == "Paris",
            "751" + df["Code du b.vote"].str[:2],
            df["Code du département"]+df["Code de la commune"].str.rjust(3,'0')
    )))
    df["CODGEO"] = cod_geo

    return df

def data_enrichment(df: pd.DataFrame) -> pd.DataFrame:

    df_mod = df.copy()

    #Formatting numerical values
    df_mod["TP6020"] = pd.to_numeric(df_mod.loc[:,"TP6020"].replace("s",None).replace("nd",None).str.replace(",","."))
    df_mod["MED20"] = pd.to_numeric(df_mod.loc[:,"MED20"].replace("s",None).replace("nd",None).str.replace(",","."))

    #Computing the population growth rate
    df_mod["POP_GROWTH_RATE"] = df_mod["P20_POP"] / df_mod["P14_POP"] - 1

    #Computing the share of principal residency
    df_mod["RP_SHARE"] = df_mod["P20_RP"] / df_mod["P20_LOG"]

    #Computing the unemployment rate
    df_mod["CHOM_RATE"] = df_mod["P20_CHOM1564"] / df_mod["P20_ACT1564"]

    #Computing the vacancy rate
    df_mod["LOGVAC_RATE"] = df_mod["P20_LOGVAC"] / df_mod["P20_LOG"]

    #For metropole (Paris, Marseille, Lyon), replacing the gps coordinate by the ones of the 1s district
    df_mod = replace_value_metropole(df_mod,'Paris','Paris 1er Arrondissement')
    df_mod = replace_value_metropole(df_mod,'Marseille','Marseille 1er Arrondissement')
    df_mod = replace_value_metropole(df_mod,'Lyon','Lyon 1er Arrondissement')

    #Dropping duplicates
    df_mod.drop_duplicates(inplace=True)

    #Computing mean and median for each numerical column
    data_mean_med = df_mod[
        ~df_mod["LIB_MOD"].isin(["Paris", "Marseille", "Lyon"]) #Excluding the metropols as it duplicates with the data of each neighborhood ('arrondissement')
    ].copy()
    data_mean_med.loc["Median"] = data_mean_med.median(numeric_only=True)
    data_mean_med.loc["Mean"] = data_mean_med.mean(numeric_only=True)
    # data_final = pd.concat([df_mod, data_mean_med.loc[["Mean","Median"]]])

    str_col = ["CODGEO","LIB_MOD", "city_code"]
    float_col = list(set(list(df_mod.columns)) - set(str_col))
    df_mod[str_col] = df_mod[str_col].astype(str)
    df_mod[float_col] = df_mod[float_col].astype(float)
    data_mean_med[str_col] = data_mean_med[str_col].astype(str)
    data_mean_med[float_col] = data_mean_med[float_col].astype(float)

    return (df_mod, data_mean_med.loc[["Mean","Median"]])

def housing_data_preproc(insee_housing_df):

    data_housing = insee_housing_df.copy()
    data_housing["CODGEO"] = np.where(
    data_housing["ARM"] == "ZZZZZ",
    data_housing["COMMUNE"],
    data_housing["ARM"]
    )

    data_housing["CODGEO"] = data_housing["CODGEO"].str.lstrip("0")

    return data_housing

def housing_cat_data_per_city(insee_housing_df_preproc, col_name, insee_housing_meta_df):

    value_to_exclude = ['YY','Y']
    insee_housing_df_preproc_subset = insee_housing_df_preproc.query(f"{col_name} not in @value_to_exclude").reset_index(drop=True)
    insee_housing_df_preproc_subset[col_name] = insee_housing_df_preproc_subset[col_name].astype(str)
    data_housing_national = pd.DataFrame(insee_housing_df_preproc_subset[col_name].value_counts(normalize=True)).reset_index()
    housing_charac = insee_housing_df_preproc_subset.groupby(["CODGEO"], as_index=False)[col_name].value_counts(normalize=True).sort_values(by=["CODGEO",col_name]).reset_index(drop=True)
    housing_charac_with_national = pd.merge(
        housing_charac,
        data_housing_national,
        how='left',
        on = col_name,
        suffixes=("_city","_national")
    
    )
    housing_charac_lib = pd.merge(
        housing_charac_with_national,
        insee_housing_meta_df.query("COD_VAR == @col_name")[["COD_MOD","LIB_MOD"]].rename(columns={"COD_MOD":col_name}),
        how='left',
        on=col_name
        )
    
    return housing_charac_lib

def housing_construction_date_df(constr_period_lib_df):

    constr_period_lib = constr_period_lib_df.copy()
    constr_period_lib["LIB_MOD"] = np.where(
        ~constr_period_lib["ACHL"].isin(["A11,A12","B11","B12","C100"]),
        "Après 2006",
        constr_period_lib["LIB_MOD"]
    )
    constr_period_lib.drop(columns=["ACHL"],inplace=True)
    constr_period_lib_proc = constr_period_lib.groupby(["CODGEO","LIB_MOD"], as_index=False)[["proportion_city","proportion_national"]].sum().reset_index(drop=True)

    return constr_period_lib_proc

def nb_rooms_bucket(nb_rooms_data):

    nb_rooms = nb_rooms_data.copy()
    nb_rooms["LIB_MOD"] = np.where(
    ~nb_rooms["NBPI"].isin(["01","02","03","04"]),
    "5 pièces et plus",
    nb_rooms["LIB_MOD"]
    )
    nb_rooms_preproc = nb_rooms.groupby(["CODGEO","LIB_MOD"], as_index=False)[["proportion_city","proportion_national"]].sum().reset_index(drop=True)
    return nb_rooms_preproc

def dvf_cleaning(dvf):

    dvf_clean = dvf.copy()
    dvf_clean["Valeur fonciere"] = dvf["Valeur fonciere"].str.replace(",",".").astype(float)
    dvf_clean["Code commune"] = dvf_clean["Code commune"].astype(str)
    dvf_clean["Surface terrain"] = dvf_clean["Surface terrain"].fillna(0)
    dvf_clean.insert(3, "CODGEO", dvf_clean["Code departement"].str.lstrip("0") + dvf_clean["Code commune"].str.rjust(3,'0'))

    #calcul de la surface habitable. Hypothèse : prise en compte de 50% du terrain pour les métropoles (Paris, Marseille, lyon), 20% sinon
    rate_metrop = 0.5
    rate_other = 0.2
    dvf_clean["Surface habitable"] = np.where(
        (dvf_clean["CODGEO"].str[:2].isin(["75","13","69"]) & (dvf_clean["CODGEO"].str.len() == 5)),
        rate_metrop * dvf_clean["Surface terrain"] + dvf_clean["Surface reelle bati"],
        rate_other * dvf_clean["Surface terrain"] + dvf_clean["Surface reelle bati"]
        )
    dvf_clean["Valeur fonc / surface habitable"] = round(dvf_clean["Valeur fonciere"] / dvf_clean["Surface habitable"], 1)
    dvf_clean["Date mutation"] = pd.to_datetime(dvf_clean["Date mutation"], format='mixed')
    dvf_clean.drop(columns=["Code departement","Code commune"], inplace=True)
    return dvf_clean

def average_dvf_price_per_city(dvf_clean_df):
    """
    Returns a dataframe with the average price per city of houses and apartments.
    """
    dvf_house_apt = dvf_clean_df.loc[
        (dvf_clean_df["Type local"].isin(["Maison", "Appartement"])) & 
        (dvf_clean_df["Nature mutation"] == "Vente")
        ].reset_index(drop=True)
    dvf_house_apt_avg = dvf_house_apt.groupby(['CODGEO','Type local'], as_index = False)["Valeur fonc / surface habitable"].mean().sort_values(by=["CODGEO","Type local"]).reset_index(drop=True)

    return dvf_house_apt_avg

def data_meteo_preproc(data_meteo):

    data_preproc = data_meteo.copy()

    #Isolating the date
    data_preproc.insert(0,"date_clean", data_preproc["date"].str[:10])
    data_preproc["date_clean"] = pd.to_datetime(data_preproc["date_clean"])

    #Dropping initial date column
    data_preproc.drop(columns=["date"], inplace=True)

    #Averaging for each day
    col_to_group = ["t","rr12", "u"]
    col_others = list(set(list(data_preproc.columns)) - set(col_to_group))
    data_preproc_avg = data_preproc.groupby(col_others, as_index=False)[col_to_group].mean().reset_index(drop=True)

    #Converting degres form Kevlin to Celsius
    data_preproc_avg["t_Celsius"] = data_preproc_avg["t"] - 273.15

    #Dropping temparature in Kelvin column
    data_preproc_avg.drop(columns=["t"], inplace=True)

    #Renaming columns
    data_preproc_avg.rename(columns={"t_Celsius":"temperature_C", "rr12":"precipitations_12h","u":"humidite_%"}, inplace = True)
    
    return data_preproc_avg

if __name__ == "__main__":

    #Decide whether exporting the data or not
    export_geographic_db = False
    export_insee_pop_data = False
    export_city_hall_elec = False
    export_presidential_elec = False
    export_housing_data = True
    export_real_estate_mkt = False
    export_meteo_data = False

    ## Output path definition
    path = os.getcwd() #get current working directory
    parent_folder = os.path.abspath(os.path.join(path, os.pardir))
    saving_folder = os.path.join(parent_folder, "app", "processed_data")

    if export_geographic_db:
        ## 1. Insee data base on city, regions etc
        #Loading data
        print("Loading data Insee cities, regions IDs...")
        db_communes = pd.read_csv("sources/commune_2022.csv", sep=",")
        db_departements = pd.read_csv("sources/departement_2022.csv", sep=",")
        db_regions = pd.read_csv("sources/region_2022.csv", sep=",")
        print("Data loaded !", "\n")

        #Merging data
        print("Merging data...")
        db_temp = pd.merge(
        db_communes,
        db_departements[["DEP","LIBELLE"]],
        how='left',
        on="DEP",
        suffixes=('_COM', '_DEP')
        ).dropna(subset=["DEP"])
        db_final = pd.merge(
            db_temp,
            db_regions[["REG","LIBELLE"]],
            how='left',
            on="REG",
            suffixes=('_COM', '_REG')
            )[["COM","DEP","LIBELLE_COM","LIBELLE_DEP", "LIBELLE"]].rename(columns={"LIBELLE": "LIBELLE_REG"})
        db_final[db_final.columns] = db_final[db_final.columns].astype(str)
        db_final["LIB_DEP"] = db_final["LIBELLE_COM"] + " (" + db_final["DEP"] + ")"
        print("Data merged !")

        #Saving data
        print("Saving data...")
        db_final.to_parquet(os.path.join(saving_folder,"cities_insee_id.parquet.gzip"), compression="gzip", index=False)
        print("Data saved !", "\n")

    if export_insee_pop_data:
        ## 2. INSEE data on population
        #Loading data
        print("Loading INSEE data...")
        data = load_insee_data(from_url=False)
        data_pop_per_city = data[0]
        meta_data_pop_per_city = data[1]
        geo_data_cities = load_cities_geo_info()
        geo_data_cities_df = load_cities_geo_info_df(geo_data_cities)
        print("INSEE data loaded!", "\n")

        #Enriching data
        print("Enriching INSEE data...")
        city_code = meta_data_pop_per_city.query("COD_VAR == 'CODGEO'")[["COD_MOD","LIB_MOD"]].drop_duplicates().rename(columns = {"COD_MOD": "CODGEO"})
        data_combined = pd.merge(city_code, data_pop_per_city, how = 'left', on = 'CODGEO')
        data_combined_geo = pd.merge(data_combined, geo_data_cities_df.rename(columns={"insee_code":"CODGEO"}), on='CODGEO', how='left')
        data_combined_enriched = data_enrichment(data_combined_geo)[0]
        data_combined_enriched["CODGEO"] = data_combined_enriched["CODGEO"].str.lstrip("0")
        data_combined_enriched_mean_med = data_enrichment(data_combined_geo)[1]
        print("INSEE data enriched", "\n")

        #saving data
        print("Saving data...")
        data_combined_enriched.to_parquet(os.path.join(saving_folder,"2020_insee_data_population.parquet.gzip"), compression="gzip", index=False)
        data_combined_enriched_mean_med.to_parquet(os.path.join(saving_folder,"2020_insee_data_population_sum_stat.parquet.gzip"), compression="gzip")
        print("Data saved !", "\n")
    
    if export_city_hall_elec:
        ## 3. Data.gouv data on city hall elections
        #Loading data
        print("Loading data on 2020 City Hall elections...")
        file_first_round_more_1000 = 'sources/2020-05-18-resultats-communes-de-1000-et-plus.xlsx'
        first_round_more_1000 = load_city_election_first_round(file_first_round_more_1000)
        file_first_round_less_1000 = 'sources/2020-05-18-resultats-communes-de-moins-de-1000.xlsx'
        first_round_less_1000 = load_city_election_first_round(file_first_round_less_1000)
        file_second_round = 'sources/election-france-municipale-2020-deuxieme-tour.csv'
        second_round_election_init = load_city_election_second_round(file_second_round)
        print("Data on 2020 City Hall elections loaded !", "\n")

        #Formatting and concatenating data
        print("Formatting and concatenating data on 2020 City Hall elections...")
        second_round_election_init_format = formatting_metropoles_votes(second_round_election_init)
        second_round_election = votes_per_city(second_round_election_init_format)
        result_election_more_1000_t1 = election_first_round_formatting(first_round_more_1000, 2, 17)
        result_election_less_1000_t1 = election_first_round_formatting(first_round_less_1000, 1, 61)
        result_election_t1 = pd.concat([result_election_more_1000_t1, result_election_less_1000_t1]).reset_index(drop=True)
        result_election_t2 = election_second_round_formatting(second_round_election)
        result_election = concat_first_second_round(result_election_t1, result_election_t2)
        result_election_format = cod_geo_result_election(result_election)
        election_dataset = election_per_city_dataset_final(result_election_format)
        election_dataset["CODGEO"] = election_dataset["CODGEO"].str.lstrip("0")
        print("Data on 2020 City Hall elections formated !", "\n")

        #saving data
        print("Saving data...")
        election_dataset.to_parquet(os.path.join(saving_folder,"2020_gouv_data_city_hall_elections.parquet.gzip"), compression="gzip", index=False)
        print("Data saved !", "\n")

    if export_presidential_elec:
        ## 4. Data.gouv data on presidential elections
        #Loading data
        print("Loading data on 2022 Presidential elections...")
        file_presidential_elec_t1 = "sources/resultats-par-niveau-burvot-t1-france-entiere.xlsx"
        file_presidential_elec_t2 = "sources/resultats-par-niveau-burvot-t2-france-entiere.xlsx"
        data_presidential_elec_t1 = load_presidential_election(file_presidential_elec_t1)
        data_presidential_elec_t2 = load_presidential_election(file_presidential_elec_t2)
        print("Data on 2022 Presidential elections loaded !", "\n")

        #Adding Geo code
        print("Formatting data on 2022 Presidential elections...")
        data_presidential_elec_cod_geo_t1 = cod_geo_result_presidential_election(data_presidential_elec_t1)
        data_presidential_elec_cod_geo_t2 = cod_geo_result_presidential_election(data_presidential_elec_t2)

        #Grouping by city for both rounds
        pres_election_per_city_t1 = presidential_election_formatting(data_presidential_elec_cod_geo_t1, 0, 11)
        pres_election_per_city_t1["Tour"] = 1
        pres_election_per_city_t2 = presidential_election_formatting(data_presidential_elec_cod_geo_t2, 0, 1)
        pres_election_per_city_t2["Tour"] = 2

        #Concatenating both rounds & formatting
        pres_election_per_city = pd.concat([pres_election_per_city_t1, pres_election_per_city_t2]).sort_values(by=["Libellé de la commune","Tour","% Voix/Exp"], ascending=[True, True, False]).reset_index(drop=True)
        str_col = ["CODGEO","Code du département", "Code de la commune","Libellé de la commune","Nom","Prénom"]
        float_col = ["Exprimés","Voix","% Voix/Exp","Tour"]
        pres_election_per_city[str_col] = pres_election_per_city[str_col].astype(str)
        pres_election_per_city[float_col] = pres_election_per_city[float_col].astype(float)
        pres_election_per_city["CODGEO"] = pres_election_per_city["CODGEO"].str.lstrip("0")
        print("Data on 2022 Presidential elections formated !", "\n")

        #saving data
        print("Saving data...")
        pres_election_per_city.to_parquet(os.path.join(saving_folder,"2022_gouv_data_pres_elections.parquet.gzip"), compression="gzip", index=False)
        print("Data saved !", "\n")

    if export_housing_data:
        print("Loading INSEE data on housing...")
        data_housing_url = "https://static.data.gouv.fr/resources/recensement-de-la-population-fichiers-detail-logements-ordinaires-en-2020-1/20231023-123618/fd-logemt-2020.parquet"
        data_housing = download_data_housing_insee(data_housing_url)
        metadata_housing_url = "sources\Dictionnaire variables_logemt_2020.csv"
        metadata_housing = pd.read_csv(metadata_housing_url, sep=";" ,low_memory=False)
        housing_data_preproc_df = housing_data_preproc(data_housing)
        print("Data loaded !", "\n")

        print("Preprocessing some data...")
        
        #Construction date
        construc_period = housing_cat_data_per_city(housing_data_preproc_df, "ACHL", metadata_housing)
        construc_period_processed = housing_construction_date_df(construc_period)

        #HLM share of housing
        hlm_housing = housing_cat_data_per_city(housing_data_preproc_df, "HLML", metadata_housing).query("HLML =='1'").reset_index(drop=True)

        #Immigration share
        immigration = housing_cat_data_per_city(housing_data_preproc_df, "IMMIM", metadata_housing)

        #Age
        age = housing_cat_data_per_city(housing_data_preproc_df, "AGEMEN8", metadata_housing)

        #Nb rooms
        nb_rooms_data = housing_cat_data_per_city(housing_data_preproc_df, "NBPI", metadata_housing)
        nb_rooms = nb_rooms_bucket(nb_rooms_data)

        #Share of home owners
        owner_share = housing_cat_data_per_city(housing_data_preproc_df, "STOCD", metadata_housing).query("STOCD == '10'").reset_index(drop=True)

        #Size (in square meters) of housing
        housing_square_meters = housing_cat_data_per_city(housing_data_preproc_df, "SURF", metadata_housing).query("SURF != 'Y'").reset_index(drop=True)
        print("Data preprocessed !", "\n")

        #Export
        print("Saving data...")
        construc_period_processed.to_parquet(os.path.join(saving_folder,"2020_housing_constr_period.parquet.gzip"), compression="gzip", index=False)
        hlm_housing.to_parquet(os.path.join(saving_folder,"2020_housing_hlm.parquet.gzip"), compression="gzip", index=False)
        immigration.to_parquet(os.path.join(saving_folder,"2020_housing_immigration.parquet.gzip"), compression="gzip", index=False)
        age.to_parquet(os.path.join(saving_folder,"2020_housing_age.parquet.gzip"), compression="gzip", index=False)
        nb_rooms.to_parquet(os.path.join(saving_folder,"2020_housing_nb_rooms.parquet.gzip"), compression="gzip", index=False)
        owner_share.to_parquet(os.path.join(saving_folder,"2020_housing_owner_share.parquet.gzip"), compression="gzip", index=False)
        housing_square_meters.to_parquet(os.path.join(saving_folder,"2020_housing_size.parquet.gzip"), compression="gzip", index=False)
        print("Data saved !", "\n")

    if export_real_estate_mkt:

        list_url = [
            "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20231010-093302/valeursfoncieres-2023.txt",
            "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20231010-093059/valeursfoncieres-2022.txt",
            "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20231010-092719/valeursfoncieres-2021.txt",
            "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20231010-092355/valeursfoncieres-2020.txt",
            "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20231010-092100/valeursfoncieres-2019.txt",
            "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20231010-091504/valeursfoncieres-2018-s2.txt"
        ]
        
        dvf_combined = pd.DataFrame()
        for url in list_url:
            print(f"Loading {url[-8:]}...")
            dvf = load_dvf(url)
            dvf_combined = pd.concat([dvf_combined, dvf])
            print("Data loaded !", "\n")

        print("Preprocessing data...")
        dvf_clean_df = dvf_cleaning(dvf_combined)
        print("Data preprocessed !", "\n")

        print("Saving data...")
        dvf_clean_df.to_parquet(os.path.join(saving_folder,"2023_real_estate_mkt.parquet.gzip"), compression="gzip", index=False)
        print("Data saved !", "\n")

    if export_meteo_data:

        print("Loading data on wheather...")
        data_meteo_2023_df = load_meteo_data("https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/donnees-synop-essentielles-omm/exports/json?lang=fr&refine=date%3A%222023%22&timezone=Europe%2FBerlin")
        print("Data loaded !", "\n")

        print("Preprocessing data...")
        data_meteo_preproc_df = data_meteo_preproc(data_meteo_2023_df)
        print("Data preprocessed !", "\n")

        print("Saving data...")
        data_meteo_preproc_df.to_parquet(os.path.join(saving_folder,"2017_2023_wheather.parquet.gzip"), compression="gzip", index=False)
        print("Data saved !", "\n")