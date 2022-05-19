import json
import argparse


def main():
    argument_parser = argparse.ArgumentParser(description='InitScript')
    argument_parser.add_argument("-rd", "--root_directory", required=True,
                                 help="project root directory")
    arguments = argument_parser.parse_args()

    root_directory = arguments.root_directory
    if root_directory == '/':
        root_directory = ''
    json_file = open("params.json", "r")
    json_object = json.load(json_file)
    json_file.close()
    json_object["dvc_root"] = root_directory
    json_file = open("params.json", "w")
    json.dump(json_object, json_file)
    json_file.close()


if __name__ == "__main__":
    main()
