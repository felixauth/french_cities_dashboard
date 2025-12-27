import os

FOLDER = "processed_data"
FILES = {
    "logo": os.path.join(FOLDER, "logo-france-2.png"),
    "insee_data": os.path.join(FOLDER,"2020_insee_data_population.parquet.gzip"),
    "insee_sum_stat": os.path.join(FOLDER,"2020_insee_data_population_sum_stat.parquet.gzip"),
    "cityhall": os.path.join(FOLDER,"2020_gouv_data_city_hall_elections.parquet.gzip"),
    "presidential": os.path.join(FOLDER,"2022_gouv_data_pres_elections.parquet.gzip"),
    "cities_insee_id": os.path.join(FOLDER,"cities_insee_id.parquet.gzip"),
    "housing_const": os.path.join(FOLDER,"2020_housing_constr_period.parquet.gzip"),
    "housing_hlm": os.path.join(FOLDER,"2020_housing_hlm.parquet.gzip"),
    "housing_immigration": os.path.join(FOLDER,"2020_housing_immigration.parquet.gzip"),
    "housing_nb_rooms": os.path.join(FOLDER,"2020_housing_nb_rooms.parquet.gzip"),
    "housing_owner_share": os.path.join(FOLDER,"2020_housing_owner_share.parquet.gzip"),
    "housing_size": os.path.join(FOLDER,"2020_housing_size.parquet.gzip"),
    "housing_age": os.path.join(FOLDER,"2020_housing_age.parquet.gzip"),
    "real_estate_2023": os.path.join(FOLDER,"2023_real_estate_mkt.parquet.gzip"),
    "weather": os.path.join(FOLDER,"2020_2023_wheather.parquet.gzip"),
    "pollution": os.path.join(FOLDER,"2023_pollution.parquet.gzip"),
    "taxes": os.path.join(FOLDER, "2022_local_taxes.parquet.gzip"),
    "rental": os.path.join(FOLDER, "2022_rental_indices.parquet.gzip"),
    "schools": os.path.join(FOLDER, "schools_data.parquet.gzip"),
}

DICT_COLOR_POL = {"Extrême Gauche": '🔴',
                "Gauche":'🟠',
                "Divers":'🟢',
                "Centre": '⚪️',
                "Droite":'🔵',
                "Extrême Droite": '⚫️'}
