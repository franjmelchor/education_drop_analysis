dvc remote modify myremote --local gdrive_service_account_json_file_path "$(pwd)"/education-drop-d884963a3b3b.json
dvc pull
PYTHONPATH="$(pwd)"/Code
export PYTHONPATH
python init.py -rd "$(pwd)"
dvc repro