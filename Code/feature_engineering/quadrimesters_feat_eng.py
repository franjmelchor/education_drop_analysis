import argparse
from apitep_utils import ArgumentParserHelper
from apitep_utils.feature_engineering import FeatureEngineering
import logging
import pandas as pd
import keys
from data_model.school_kind import SchoolKind

log = logging.getLogger(__name__)

pr_plan_subject_call: pd.DataFrame
pr_scholarship_per_year: pd.DataFrame


def get_cum_p_data_plan_subj_call(p: pd.Series, course: int, quadrimester: int):
    global pr_plan_subject_call
    p_data = pr_plan_subject_call[(pr_plan_subject_call['cod_plan'] == p.cod_plan) &
                                  (pr_plan_subject_call['expediente'] == p.expediente)
                                  ].sort_values(by=['curso_academico'])
    if len(p_data.index) > 0:
        courses = p_data['curso_academico'].unique()[0:course]

        p_data = p_data[p_data['curso_academico'].isin(courses)]
        if quadrimester == 1:
            try:
                quadrimester_data = p_data[(p_data['curso_academico'] == courses[course - 1]) &
                                           (p_data['semestre'] == '1S')].index
            except:
                return pd.Series()
        elif quadrimester == 2:
            try:
                quadrimester_data = p_data[(p_data['curso_academico'] == courses[course - 1]) &
                                           (p_data['semestre'] == '2S')].index
            except:
                return pd.Series()
        else:
            raise NotImplementedError

        if len(quadrimester_data > 0):
            if quadrimester == 1:
                no_valid_data = p_data[(p_data['curso_academico'] == courses[course - 1]) &
                                       (p_data['semestre'] == '2S')].index
                if len(p_data.index > 0):
                    p_data = p_data[p_data.index.isin(no_valid_data) == False]
        else:
            return pd.Series()

    return p_data


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


def get_cum_pass_ratio(p: pd.Series, course: int, quadrimester: int):
    p_data = get_cum_p_data_plan_subj_call(p, course, quadrimester)
    if len(p_data.index) > 0:
        n_subjects = len(p_data.index)
        n_passed_subjects = len(p_data[p_data['nota_num'] >= 5.0].index)
        if n_subjects > 0:
            return n_passed_subjects / n_subjects
        else:
            return -1
    else:
        return -1


def get_scholarship(p: pd.Series, course: int):
    p_data = get_course_p_data_scholarship(p, course)
    if len(p_data.index) > 0:
        return p_data['becario'].values[0]
    else:
        return -1


def get_cum_absent_ratio(p: pd.Series, course: int, quadrimester: int):
    p_data = get_cum_p_data_plan_subj_call(p, course, quadrimester)
    if len(p_data.index > 0):
        n_subjects = len(p_data.index)
        n_absent_subjects = len(p_data[p_data['des_nota']=='NO PRESENTADO'].index)

        return n_absent_subjects / n_subjects
    else:
        return -1


def get_cum_median(p: pd.Series, course: int, quadrimester: int):
    import numpy as np
    p_data = get_cum_p_data_plan_subj_call(p, course, quadrimester)
    if len(p_data.index) > 0:
        return np.nanmedian(p_data['nota_num'])
    else:
        return -1


def get_cum_more_1st_call_ratio(p: pd.Series, course: int, quadrimester: int):
    p_data = get_cum_p_data_plan_subj_call(p, course, quadrimester)

    if len(p_data.index) > 0:
        n_more_1st_call_subjects = len(p_data[p_data['numero_convocatorias_agotadas'] > 1])
        n_subjects = len(p_data.index)

        if n_subjects > 0:
            return n_more_1st_call_subjects / n_subjects
        else:
            return -1
    else:
        return -1


class QuadrimestersFeatureEngineering(FeatureEngineering):
    quadrimester: int = None
    course: int = None
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

        program_description = self.description
        argument_parser = argparse.ArgumentParser(description=program_description)
        argument_parser.add_argument("-i", "--input_paths",
                                     required=True,
                                     nargs="+",
                                     help="path to the input CSV datasets")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        argument_parser.add_argument("-s", "--school_kind", required=True,
                                     help="school kind to analyze")
        argument_parser.add_argument("-c", "--course", required=True,
                                     help="course to analyze")
        argument_parser.add_argument("-q", "--quadrimester", required=True,
                                     help="quadrimester of course to analyze")

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
        self.course = int(arguments.course)
        self.quadrimester = int(arguments.quadrimester)
        school_kind_str = arguments.school_kind
        if school_kind_str == "Teaching":
            self.school_kind = SchoolKind.Teaching
        elif school_kind_str == "Polytechnic":
            self.school_kind = SchoolKind.Polytechnic

    @FeatureEngineering.stopwatch
    def process(self):
        """
        Feature Engineering of pred_analys_record_personal_access
        """

        log.info("Feature Engineering of pred_analys_record_personal_access data of school: " + self.school_kind.value)
        log.debug("QuadrimestersFeatureEngineering.process()")

        global pr_plan_subject_call, pr_scholarship_per_year

        analys_record_personal_access = self.input_dfs[0]
        pr_plan_subject_call = self.input_dfs[1]
        pr_scholarship_per_year = self.input_dfs[2]

        cols_before = len(self.input_dfs[0].columns)
        col_list_before = self.input_dfs[0].columns

        analys_record_personal_access[keys.CUM_PASS_RATIO_KEY] = analys_record_personal_access.apply(
            lambda func: get_cum_pass_ratio(func, course=self.course, quadrimester=self.quadrimester), axis=1
        )
        analys_record_personal_access[keys.SCHOLARSHIP_KEY] = analys_record_personal_access.apply(
            lambda func: get_scholarship(func, course=self.course), axis=1
        )
        analys_record_personal_access[keys.CUM_ABSENT_RATIO_KEY] = analys_record_personal_access.apply(
            lambda func: get_cum_absent_ratio(func, course=self.course, quadrimester=self.quadrimester), axis=1
        )
        analys_record_personal_access[keys.CUM_MEDIAN_KEY] = analys_record_personal_access.apply(
            lambda func: get_cum_median(func, course=self.course, quadrimester=self.quadrimester), axis=1
        )

        if self.course > 1:
            analys_record_personal_access[keys.CUM_MORE_1ST_CALL_RATIO_KEY] = analys_record_personal_access.apply(
                lambda func: get_cum_more_1st_call_ratio(func, course=self.course, quadrimester=self.quadrimester),
                axis=1
            )
        cols_after = len(self.input_dfs[0].columns)
        col_list_after = self.input_dfs[0].columns
        self.changes["add new columns with information of quadrimester"] = cols_after - cols_before
        log.info("new columns are :" + str(list(set(col_list_after) - set(col_list_before))))
        log.info("final columns are: " + str(col_list_after))

        rows_before = len(analys_record_personal_access.index)
        analys_record_personal_access = analys_record_personal_access[analys_record_personal_access
                                                                      [keys.CUM_PASS_RATIO_KEY] != -1]
        rows_after = len(analys_record_personal_access.index)

        self.changes["removes people who have dropped out in this quadrimester"] = rows_before - rows_after
        self.changes["number of final rows"] = rows_after

        self.output_df = analys_record_personal_access


def main():
    logging.basicConfig(
        filename="quadrimesters_debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start RecordPersonalAccessETL")
    log.debug("main()")

    feat_eng = QuadrimestersFeatureEngineering(
        input_separator='|',
        output_separator='|',
        save_report_on_load=False,
        save_report_on_save=False
    )
    feat_eng.execute()


if __name__ == "__main__":
    main()
