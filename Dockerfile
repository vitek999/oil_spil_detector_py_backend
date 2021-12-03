FROM python:3.8.12

WORKDIR /app

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install Flask
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install sentinelhub
RUN pip install numpy
RUN pip install pickle-mixin

COPY mai.py /app/app.py
COPY pickle_model.pkl /app/pickle_model.pkl

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080"]