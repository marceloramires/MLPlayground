FROM python:3.11

#COPY surname_nationality_classifying_service.py surname_nationality_classifying_service.py
# Copy all applicaiton files
COPY . .
RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install flask gunicorn matplotlib flask_cors --no-cache-dir

ENTRYPOINT ["gunicorn", "surname_nationality_classifying_service:app", "run", "--bind", "0.0.0.0:80"]