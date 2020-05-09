This project is the backend API for online explaining a given piece of code

## requirements
python 3.6, django, django REST framework, keras 2.2.4, tensorflow 1.14

## setup
- download model from https://s3.us-east-2.amazonaws.com/icse2018/funcom.tar.gz
- setup api/views.py
  - set variable 'modelfile' to the path to 'standard_attend-gru_E04_TA0.72_VA0.66.h5' model
  - set variables 'xxxtokFile' to the path to the dataset tokenfile
- run python manage.py runserver 0.0.0.0:\[portNumber\]