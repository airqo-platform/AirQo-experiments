FROM ubuntu:16.04

# Install baseimage libraries 
RUN apt-get update -y && \
    apt-get install -y python3.5-pip python3.5-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Create app directory
WORKDIR /app
# Install app dependencies 
RUN pip install -r requirements.txt
# Bundle the app source
COPY . /app
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]