FROM public.ecr.aws/lambda/python:3.9

RUN yum -y update
RUN yum install -y mesa-libGL
RUN yum -y install gcc
RUN yum install -y git

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

ENV PATH="${PATH}:/root/.poetry/bin"

COPY pyproject.toml ./
COPY poetry.lock ./

RUN poetry install

# ARG AWS_ACCESS_KEY_ID_ARG
# ARG AWS_SECRET_ACCESS_KEY_ARG
# ARG AWS_DEFAULT_REGION_ARG

# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_ARG
# ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_ARG
# ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION_ARG

# Copy model checkpoint into Docker
RUN mkdir model_dir
COPY data/BERT_ft_epoch5.model ./model_dir

# copy the training script inside the container

COPY web_app/app.py ./ 

CMD ["app.handler"]