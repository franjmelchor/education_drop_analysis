import argparse
from apitep_utils import ArgumentParserHelper
from apitep_utils.feature_engineering import FeatureEngineering
import logging

import keys
import numpy as np
import pandas as pd

from data_model.school_kind import SchoolKind

log = logging.getLogger(__name__)


class RecordPersonalAccessFeatureEngineering(FeatureEngineering):
    school_kind: SchoolKind = None

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
        argument_parser.add_argument("-s", "--school_kind", required=True,
                                     help="school kind to analyze")

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
        school_kind_str = arguments.school_kind
        if school_kind_str == "Teaching":
            self.school_kind = SchoolKind.Teaching
        elif school_kind_str == "Polytechnic":
            self.school_kind = SchoolKind.Polytechnic

    @FeatureEngineering.stopwatch
    def process(self):
        """
        Feature Engineering int_record_personal_access
        """

        log.info("Feature Engineering of pr_record_personal_access data of school: " + self.school_kind.value)
        log.debug("RecordPersonalAccessFeatureEngineering.process()")

        analys_columns = [keys.RECORD_KEY, keys.PLAN_CODE_KEY, keys.PLAN_DESCRIPTION_KEY, keys.OPEN_YEAR_PLAN_KEY,
                          keys.DROP_OUT_KEY, keys.ACCESS_CALL_KEY, keys.ACCESS_DESCRIPTION_KEY,
                          keys.FINAL_ADMISION_NOTE_KEY, keys.GENDER_KEY, keys.BIRTH_DATE_KEY,
                          keys.PROVINCE_KEY, keys.TOWN_KEY]

        cols_before = len(self.input_dfs[0].columns)
        col_list_before = self.input_dfs[0].columns
        self.input_dfs[0] = self.input_dfs[0][analys_columns]
        cols_after = len(self.input_dfs[0].columns)
        col_list_after = self.input_dfs[0].columns
        self.changes["delete columns unused to analysis"] = cols_before - cols_after
        log.info("deleted columns are :" + str(list(set(col_list_before) - set(col_list_after))))
        log.info("final columns are: " + str(analys_columns))

        self.input_dfs[0].dropna(inplace=True)

        self.input_dfs[0][keys.BIRTH_DATE_KEY] = pd.to_datetime(self.input_dfs[0][keys.BIRTH_DATE_KEY])
        self.input_dfs[0][keys.BIRTH_DATE_KEY] = self.input_dfs[0][keys.BIRTH_DATE_KEY].apply(lambda func: func.year)
        self.input_dfs[0].rename(columns={keys.BIRTH_DATE_KEY: keys.BIRTH_YEAR_KEY}, inplace=True)
        anio_nacimiento_bcket_array = np.linspace(1960, 2005, 10)
        self.input_dfs[0][keys.BIRTH_YEAR_INTERVAL_KEY] = pd.cut(
            self.input_dfs[0][keys.BIRTH_YEAR_KEY], anio_nacimiento_bcket_array, include_lowest=True)
        self.input_dfs[0].drop([keys.BIRTH_YEAR_KEY], axis=1, inplace=True)
        log.info("Get only birth year of birth date")

        self.input_dfs[0][keys.DROP_OUT_KEY] = self.input_dfs[0][keys.DROP_OUT_KEY].apply(
            lambda func: 1 if func == 'S' else 0)
        log.info("Change format to " + keys.DROP_OUT_KEY + " feature")

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
