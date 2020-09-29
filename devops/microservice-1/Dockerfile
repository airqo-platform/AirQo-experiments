# python runtime
FROM python:3.6.5-alpine

# working directory
WORKDIR /app

# copy current directory into the container
ADD . /app

# install requirements
RUN pip install -r "devops/microservice-1/requirements.txt"

# make port 5000 available to the world outside
EXPOSE 5000

#CMD ["gunicorn", "--config", "./conf/gunicorn_conf.py", "src:app"]
# ENTRYPOINT ["python"]
CMD ["python3", "app.py"]