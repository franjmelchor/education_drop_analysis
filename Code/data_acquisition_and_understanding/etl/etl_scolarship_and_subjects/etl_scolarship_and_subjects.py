import logging
from apitep_utils import ETL
import sys
import os
sys.path.append(os.path.relpath('./'))
from apitep_utils import ArgumentParserHelper
import argparse
import sys

print(sys.path)
log = logging.getLogger(__name__)

PLAN_DESCRIPTION_KEY = "des_plan"
class ScholarshipPerYearAndPlanSubjCallETL(ETL):

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

        program_description = self.description
        argument_parser = argparse.ArgumentParser(description=program_description)
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

    @ETL.stopwatch
    def process(self):
        """
        Process scholarship_per_year data
        """

        log.info("Process scholarship_per_year and plan_subject_call data")
        log.debug("ScholarshipPerYearAndPlanSubjCallETL.process()")

        old_degrees = ['MÁSTER UNIVERSITARIO EN COMPUTACIÓN GRID Y PARALELISMO',
                       'MÁSTER DE ESPECIALIZACIÓN EN GEOTECNOLOGÍAS TOPOGRÁFICAS EN LA INGENIERÍA',
                       'MÁSTER UNIVERSITARIO EN EVALUACIÓN Y GESTIÓN DEL RUIDO AMBIENTAL',
                       'MÁSTER EN COMPUTACIÓN GRID Y PARALELISMO']

        rows_before = len(self.input_dfs[0].index)
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0][PLAN_DESCRIPTION_KEY].isin(old_degrees) == False]
        rows_after = len(self.input_dfs[0].index)
        self.changes["delete data of old degrees"] = rows_before - rows_after

        log.info("columns of final dataset are:" + self.input_dfs[0].columns)

        self.output_df = self.input_dfs[0]


def main():
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start ScholarshipPerYearAndPlanSubjCallETL")
    log.debug("main()")

    etl = ScholarshipPerYearAndPlanSubjCallETL(
        input_separator="|",
        output_separator="|",
        save_report_on_load=False,
        save_report_on_save=False,
        report_type=ETL.ReportType.Standard,
    )
    etl.execute()


if __name__ == "__main__":
    main()
