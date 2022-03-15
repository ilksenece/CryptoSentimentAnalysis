# syntax=docker/dockerfile:1

# Use latest Python runtime as a parent image
FROM python:3.8-slim-buster

# Meta-data
LABEL maintainer="Ece Ay <eceicyuz@gmail.com>" \
      description="Docker Crypto Sentiment Analysis Application\
      This specific use case has the container set up like an executible.\
      Code plus dependencies required to run the program are installed in the\
      container. Data are set up via a mounted folder. Executing\
      docker run with a parameter starts up a dashboard"

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# pip install
RUN pip3 install -r requirements.txt

# Make port available to the world outside this container
EXPOSE 5000

# ENTRYPOINT allows us to specify the default executible
ENTRYPOINT ["python", "app.py"]

# CMD sets default arguments to executable which may be overwritten when using docker run
CMD ["python", "app.py", "--host=0.0.0.0"]
