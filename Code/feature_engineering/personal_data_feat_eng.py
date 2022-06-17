import argparse
from apitep_utils import ArgumentParserHelper
from apitep_utils.feature_engineering import FeatureEngineering
import logging

import keys
import numpy as np
import pandas as pd

import statistics

log = logging.getLogger(__name__)

dset = pd.DataFrame
pr_scholarship_per_year: pd.DataFrame


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

        analys_columns = [keys.RECORD_KEY, keys.PLAN_CODE_KEY, keys.PLAN_DESCRIPTION_KEY, keys.OPEN_YEAR_PLAN_KEY,
                          keys.DROP_OUT_KEY, keys.ACCESS_CALL_KEY, keys.ACCESS_DESCRIPTION_KEY,
                          keys.FINAL_ADMISION_NOTE_KEY, keys.GENDER_KEY, keys.BIRTH_DATE_KEY,
                          keys.PROVINCE_KEY, keys.TOWN_KEY]

        global dset, pr_scholarship_per_year

        dset = self.input_dfs[0].copy()
        pr_scholarship_per_year = self.input_dfs[1]

        cols_before = len(self.input_dfs[0].columns)
        self.input_dfs[0] = self.input_dfs[0][analys_columns]
        cols_after = len(self.input_dfs[0].columns)
        self.changes["delete columns unused to analysis"] = cols_before - cols_after
        log.info("final columns are: " + str(analys_columns))

        null_values_before = self.input_dfs[0].isnull().sum().sum()
        self.input_dfs[0][keys.FINAL_ADMISION_NOTE_KEY] = self.input_dfs[0].apply(
            lambda func: get_median(func.cod_plan) if pd.isna(func[keys.FINAL_ADMISION_NOTE_KEY]) else func[
                keys.FINAL_ADMISION_NOTE_KEY], axis=1)
        null_values_after = self.input_dfs[0].isnull().sum().sum()
        self.changes["resolve null values of " + keys.FINAL_ADMISION_NOTE_KEY] = null_values_before - null_values_after

        col_list_before = self.input_dfs[0].columns
        self.input_dfs[0]['lugar_origen'] = self.input_dfs[0][keys.TOWN_KEY].apply(
            lambda func: 'MISMO_MUNICIPIO' if func == 'CÁCERES' else (func if pd.isna(func) else 'OTRO_MUNICIPIO'))
        self.input_dfs[0]['lugar_origen'] = self.input_dfs[0].apply(
            lambda func: func.lugar_origen if func[keys.PROVINCE_KEY] == 'CÁCERES' or
                                              func[keys.PROVINCE_KEY] == 'BADAJOZ' else (
                func[keys.PROVINCE_KEY] if pd.isna(func[keys.PROVINCE_KEY]) else 'OTRA_COMUNIDAD'), axis=1)
        self.input_dfs[0].drop([keys.TOWN_KEY, keys.PROVINCE_KEY], axis=1, inplace=True)
        col_list_after = self.input_dfs[0].columns
        log.info("deleted columns are :" + str(list(set(col_list_before) - set(col_list_after))))
        log.info("new column is: lugar_origen")
        log.info("final columns are: " + str(analys_columns))

        rows_before = len(self.input_dfs[0].index)
        self.input_dfs[0].dropna(inplace=True)
        rows_after = len(self.input_dfs[0].index)
        self.changes["remove nan values"] = rows_before - rows_after

        self.input_dfs[0]['fecha_curso'] = '2016-01-01'
        self.input_dfs[0]['fecha_curso'] = pd.to_datetime(self.input_dfs[0]['fecha_curso'])
        self.input_dfs[0]['fecha_nacimiento'] = pd.to_datetime(self.input_dfs[0]['fecha_nacimiento'])
        self.input_dfs[0]['edad_acceso'] = self.input_dfs[0].apply(
            lambda func: func.fecha_curso.year - func.fecha_nacimiento.year, axis=1)
        year_bcket_array = np.array([18, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        self.input_dfs[0]['edad_acceso'] = pd.cut(
            self.input_dfs[0]['edad_acceso'], year_bcket_array, include_lowest=True)
        self.input_dfs[0].drop([keys.BIRTH_DATE_KEY, 'fecha_curso'], axis=1, inplace=True)

        self.input_dfs[0][keys.DROP_OUT_KEY] = self.input_dfs[0][keys.DROP_OUT_KEY].apply(
            lambda func: 1 if func == 'S' else 0)
        log.info("Change format to " + keys.DROP_OUT_KEY + " feature")

        self.input_dfs[0][keys.SCHOLARSHIP_KEY] = self.input_dfs[0].apply(
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
        save_report_on_save=True
    )
    feature_eng.execute()


if __name__ == "__main__":
    main()
