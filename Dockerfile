FROM python:3.8
WORKDIR /cement_strenght_prediction
COPY . /cement_strenght_prediction/
RUN pip install -r requirement.txt
EXPOSE 3000
CMD python ./src/run.py --action predict