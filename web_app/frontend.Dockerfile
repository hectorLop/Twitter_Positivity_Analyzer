FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip

RUN pip install gradio && \
    pip install boto3

ARG AWS_ACCESS_KEY_ID_ARG
ARG AWS_SECRET_ACCESS_KEY_ARG
ARG AWS_DEFAULT_REGION_ARG

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_ARG
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_ARG
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION_ARG

# copy the scripts inside the container
COPY web_app/frontend.py ./
COPY twitter_analyzer ./

EXPOSE 8500

ENTRYPOINT ["python3","-m", "frontend"]