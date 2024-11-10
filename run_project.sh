dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote --local gdrive_service_account_json_file_path "$(pwd)"/education-drop-4e7427aa22db.json
dvc pull
PYTHONPATH="$(pwd)"/Code
export PYTHONPATH
python3 init.py -rd "$(pwd)"
dvc repro