# CSV Columns Keys
PLAN_DESCRIPTION_KEY = "des_plan"
RECORD_KEY = "expediente"
PLAN_CODE_KEY = "cod_plan"
PLAN_CODE_RECORD_KEY = "cod_plan_exp"
AGE_KEY = "edad_acceso"
BIRTH_DATE_KEY = "fecha_nacimiento"
BIRTH_YEAR_KEY = "anio_nacimiento"
BIRTH_YEAR_INTERVAL_KEY = "anio_nacimiento_interval"
OPEN_YEAR_PLAN_KEY = "anio_apertura_expediente"
OPEN_DATE_PLAN_KEY = "fecha_apertura_expediente"
ACCESS_CALL_KEY = "convocatoria_acceso"
ACCESS_DESCRIPTION_KEY = "des_acceso"
GENDER_KEY = "sexo"
PROVINCE_KEY = "provincia"
TOWN_KEY = "municipio"
TRANSFER_TYPE_KEY = "tipo_traslado"
CUM_PASS_RATIO_KEY = "cum_pass_ratio"
SCHOLARSHIP_KEY = "becario"
CUM_ABSENT_RATIO_KEY = "cum_absent_ratio"
CUM_MEDIAN_KEY = "cum_median"
CUM_PASS_MEDIAN_KEY = "cum_pass_median"
CUM_FAIL_MEDIAN_KEY = "cum_fail_median"
STD_DEVIATION_KEY = "std_deviation"
CUM_MEDIAN_INTERVAL_KEY = "cum_median_interval"
CUM_MORE_1ST_CALL_RATIO_KEY = "cum_more_1st_call_ratio"
FINAL_ADMISION_NOTE_KEY = "nota_admision_def"
FINAL_ADMISION_NOTE_INTERVAL_KEY = "nota_admision_def_interval"
DROP_OUT_KEY = "abandona"
FST_MODEL_KEY = "1st_model"
SND_MODEL_KEY = "2nd_model"
TRD_MODEL_KEY = "3rd_model"
FTH_MODEL_KEY = "4th_model"
FITH_MODEL_KEY = "5th_model"
SITH_MODEL_KEY = "6th_model"
SETH_MODEL_KEY = "7th_model"
ETH_MODEL_KEY = "8th_model"
NTH_MODEL_KEY = "9th_model"
TRAIN_KEY = "train"
DISTANCE_KEY = "distance"



import argparse
from apitep_utils import ArgumentParserHelper
from apitep_utils.feature_engineering import FeatureEngineering
import logging

import pandas as pd

import statistics
import multiprocessing

log = logging.getLogger(__name__)

dset = pd.DataFrame
pr_scholarship_per_year: pd.DataFrame


def get_distance(city_origin, city_target):
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    import requests
    import json
    geolocator = Nominatim(user_agent='edrop_cities')
    location_origin = RateLimiter(geolocator.geocode, min_delay_seconds=1)(city_origin)
    location_target = RateLimiter(geolocator.geocode, min_delay_seconds=1)(city_target)
    distance = -1
    if location_origin is not None and location_target is not None:
        r = requests.get(f"http://router.project-osrm.org/route/v1/car/{location_origin.longitude},"
                         f"{location_origin.latitude};{location_target.longitude},{location_target.latitude}?overview=false""")
        routes = json.loads(r.content)
        if 'España' in str(location_target):
            distance = routes.get("routes")[0]['distance'] / 1000
        elif 'MEDELLÍN' in city_target:
            distance = 114
        elif 'SANTA MARTA' in city_target:
            distance = 120
        elif 'GUADALUPE' in city_target:
            distance = 152
        elif 'ATALAYA' in city_target:
            distance = 97.4
        elif 'HELECHAL' in city_target:
            distance = 180
        elif 'RIO TURBIO' in city_target:
            distance = 503
        elif 'VALDIVIA' in city_target:
            distance = 93
        elif 'VALVERDE' in city_target:
            distance = 1892
        elif 'CARTAGENA' in city_target:
            distance = 699
        elif 'GUADALAJARA' in city_target:
            distance = 358
        elif 'ENTRERRÍOS' in city_target:
            distance = 98
    return distance


def get_median(cod_plan):
    global dset
    return statistics.median(
        dset['nota_admision_def'][(dset['nota_admision_def'].notna()) & (dset['cod_plan'] == cod_plan)])


def get_course_p_data_scholarship(p: pd.Series, course: int):
    p_data = pr_scholarship_per_year[(pr_scholarship_per_year['cod_plan'] == p.cod_plan)
                                     & (pr_scholarship_per_year['expediente'] == p.expediente)
                                     ].sort_values(by=['curso_academico'])
    try:
        academic_year = p_data['curso_academico'].unique()[course - 1]
        p_data = p_data[p_data['curso_academico'] == academic_year]
        return p_data
    except:
        return pd.Series()


def get_scholarship(p: pd.Series, course: int):
    p_data = get_course_p_data_scholarship(p, course)
    if len(p_data.index) > 0:
        return p_data['becario'].values[0]
    else:
        return -1


class RecordPersonalAccessFeatureEngineering(FeatureEngineering):

    def parse_arguments(self):
        """
        Parse arguments provided via command line, and check if they are valid
        or not. Adequate defaults are provided when possible.

        Parsed arguments are:
        - paths to the input CSV datasets, separated with spaces.
        - path to the output CSV dataset.
        """

        log.info("Get integration arguments")
        log.debug("Integration.parse_arguments()")

        argument_parser = argparse.ArgumentParser(description=self.description)
        argument_parser.add_argument("-i", "--input_paths",
                                     required=True,
                                     nargs="+",
                                     help="path to the input CSV datasets")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        arguments = argument_parser.parse_args()
        input_path_segments = arguments.input_paths
        self.input_path_segments = []
        for input_path_segment in input_path_segments:
            self.input_path_segments.append(
                ArgumentParserHelper.parse_data_file_path(
                    data_file_path=input_path_segment)
            )
        self.output_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.output_path,
            check_is_file=False)

    @FeatureEngineering.stopwatch
    def process(self):
        """
        Feature Engineering int_record_personal_access
        """

        log.info("Feature Engineering of pr_record_personal_access data")
        log.debug("RecordPersonalAccessFeatureEngineering.process()")
        log.info("initial columns are: " + str(self.input_dfs[0].columns))

        analys_columns = [RECORD_KEY, PLAN_CODE_KEY, PLAN_DESCRIPTION_KEY, OPEN_YEAR_PLAN_KEY,
                          DROP_OUT_KEY, ACCESS_CALL_KEY, ACCESS_DESCRIPTION_KEY,
                          FINAL_ADMISION_NOTE_KEY, GENDER_KEY, BIRTH_DATE_KEY,
                          PROVINCE_KEY, TOWN_KEY]

        global dset, pr_scholarship_per_year

        dset = self.input_dfs[0].copy()
        pr_scholarship_per_year = self.input_dfs[1]

        cols_before_names = self.input_dfs[0].columns
        cols_before = len(self.input_dfs[0].columns)
        self.input_dfs[0] = self.input_dfs[0][analys_columns]
        cols_after = len(self.input_dfs[0].columns)
        self.changes["delete columns unused to analysis"] = cols_before - cols_after
        log.info("final columns are: " + str(analys_columns))
        log.info("deleted columns are: " + str(set(cols_before_names) - set(analys_columns)))

        null_values_before = self.input_dfs[0].isnull().sum().sum()
        self.input_dfs[0][FINAL_ADMISION_NOTE_KEY] = self.input_dfs[0].apply(
            lambda func: get_median(func.cod_plan) if pd.isna(func[FINAL_ADMISION_NOTE_KEY]) else func[
                FINAL_ADMISION_NOTE_KEY], axis=1)
        null_values_after = self.input_dfs[0].isnull().sum().sum()
        self.changes["resolve null values of " + FINAL_ADMISION_NOTE_KEY] = null_values_before - null_values_after

        rows_before = len(self.input_dfs[0].index)
        self.input_dfs[0].dropna(inplace=True)
        rows_after = len(self.input_dfs[0].index)
        self.changes["remove nan values"] = rows_before - rows_after

        self.input_dfs[0]['fecha_curso'] = '2016-01-01'
        self.input_dfs[0]['fecha_curso'] = pd.to_datetime(self.input_dfs[0]['fecha_curso'])
        self.input_dfs[0]['fecha_nacimiento'] = pd.to_datetime(self.input_dfs[0]['fecha_nacimiento'])
        self.input_dfs[0]['edad_acceso'] = self.input_dfs[0].apply(
            lambda func: func.fecha_curso.year - func.fecha_nacimiento.year, axis=1)

        self.input_dfs[0].drop([BIRTH_DATE_KEY, 'fecha_curso'], axis=1, inplace=True)

        self.input_dfs[0]['distance'] = self.input_dfs[0]['municipio'].swifter.set_npartitions(
            multiprocessing.cpu_count()).apply(lambda func: get_distance('CACERES', func))
        self.input_dfs[0].drop([TOWN_KEY, PROVINCE_KEY], axis=1, inplace=True)
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0]['distance'] != -1]

        self.input_dfs[0][DROP_OUT_KEY] = self.input_dfs[0][DROP_OUT_KEY].apply(
            lambda func: 1 if func == 'S' else 0)
        log.info("Change format to " + DROP_OUT_KEY + " feature")

        self.input_dfs[0][SCHOLARSHIP_KEY] = self.input_dfs[0].apply(
            lambda func: get_scholarship(func, course=1), axis=1
        )

        self.output_df = self.input_dfs[0]


def main():
    logging.basicConfig(
        filename="personal_access_debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start RecordPersonalAccessFeatureEngineering")
    log.debug("main()")

    feature_eng = RecordPersonalAccessFeatureEngineering(
        input_separator='|',
        output_separator='|',
        report_type=FeatureEngineering.ReportType.Standard,
        save_report_on_load=False,
        save_report_on_save=False
    )
    feature_eng.execute()


if __name__ == "__main__":
    main()
