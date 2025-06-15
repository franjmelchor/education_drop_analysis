import logging
from apitep_utils import ETL
import sys
import os
sys.path.append(os.path.relpath('./'))
import pandas as pd
from apitep_utils import ArgumentParserHelper
import argparse

log = logging.getLogger(__name__)
pr_plan_subj_call: pd.DataFrame
PLAN_DESCRIPTION_KEY = "des_plan"
OPEN_YEAR_PLAN_KEY = "anio_apertura_expediente"


def delete_people_without_all_information(p):
    global pr_plan_subj_call

    p_data = pr_plan_subj_call[(pr_plan_subj_call['cod_plan'] == p.cod_plan) &
                               (pr_plan_subj_call['expediente'] == p.expediente)]
    if len(p_data[(p_data['tipologia_asignatura'] == 'T')
                  & (p_data['curso_asignatura'] == 1)
                  & (p_data['numero_convocatorias_agotadas'] <= 1)].index) < 10:
        return False
    else:
        return True


class RecordPersonalAccessETL(ETL):

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
        Process record_personal_access data
        """

        log.info("Process record_personal_access data ")
        log.debug("RecordPersonalAccessETL.process()")

        global pr_plan_subj_call

        old_degrees = ['MÁSTER UNIVERSITARIO EN COMPUTACIÓN GRID Y PARALELISMO',
                       'MÁSTER DE ESPECIALIZACIÓN EN GEOTECNOLOGÍAS TOPOGRÁFICAS EN LA INGENIERÍA',
                       'MÁSTER UNIVERSITARIO EN EVALUACIÓN Y GESTIÓN DEL RUIDO AMBIENTAL',
                       'MÁSTER EN COMPUTACIÓN GRID Y PARALELISMO']

        no_valid_courses = ['2017-18', '2018-19', '2019-20', '2020-21']
        pr_plan_subj_call = self.input_dfs[1]

        log.info("initial number of rows: " + str(len(self.input_dfs[0].index)))
        rows_before = len(self.input_dfs[0].index)
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0][PLAN_DESCRIPTION_KEY].isin(old_degrees) == False]
        rows_after = len(self.input_dfs[0].index)
        self.changes["delete data of old degrees"] = rows_before - rows_after

        rows_before = len(self.input_dfs[0].index)
        df_filter = self.input_dfs[0].apply(delete_people_without_all_information, axis=1)
        self.input_dfs[0] = self.input_dfs[0][df_filter]
        self.input_dfs[0] = self.input_dfs[0][self.input_dfs[0][OPEN_YEAR_PLAN_KEY].isin(no_valid_courses)
                                              == False]
        rows_after = len(self.input_dfs[0].index)
        self.changes["delete data of people without all information"] = rows_before - rows_after

        log.info("columns of final dataset are:" + str(self.input_dfs[0].columns))
        log.info("final number of rows: " + str(len(self.input_dfs[0].index)))

        self.output_df = self.input_dfs[0]


def main():

    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start RecordPersonalAccessETL")
    log.debug("main()")

    etl = RecordPersonalAccessETL(
        input_separator="|",
        output_separator="|",
        save_report_on_save=True,
        save_report_on_load=True,
        report_type=ETL.ReportType.Advanced,
    )
    etl.execute()


if __name__ == "__main__":
    main()
