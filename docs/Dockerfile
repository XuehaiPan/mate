FROM sphinxdoc/sphinx:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install --quiet --yes --no-install-recommends \
  texlive-latex-recommended texlive-latex-extra texlive-pictures pdf2svg && \
  rm -rf /var/lib/apt/list/*

COPY requirements.txt /mate/requirements.txt
COPY docs/requirements.txt /mate/docs/requirements.txt
RUN pip3 install -r /mate/requirements.txt -r /mate/docs/requirements.txt && \
  rm -rf /root/.cache

COPY . /mate
WORKDIR /mate

EXPOSE 8000
ENTRYPOINT sphinx-autobuild --host 0.0.0.0 --port 8000 --watch . docs/source /tmp/mate/docs
