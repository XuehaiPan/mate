# The Multi-Agent Tracking Environment's Documentation

This directory contains the documentation of `MATE`, the ***M**ulti-**A**gent **T**racking **E**nvironment*.

### Requirements

- `sphinx`
- `sphinx-autobuild`
- `sphinx-rtd-theme`
- `sphinxcontrib-tikz`
- `make`
- `pdflatex`
- `pdf2svg` / `ghostscript` / `pdftoppm + pnmtopng`

### Steps to build the documentation

```bash
cd docs  # navigate to this directory
pip3 install -r requirements.txt
sphinx-autobuild --watch ../mate --open-browser source build
```

A `pdflatex` installation is required to generate the TikZ pictures. If you haven't install a TeX distribution on your device yet, you can build the documentation with [docker](https://www.docker.com/get-started):

```bash
cd docs/..  # navigate to the project root directory
docker build --file docs/Dockerfile --tag mate-docs .
docker run -it --rm -p 8000:8000 mate-docs
```

Then you can browse the documentation at http://localhost:8000. Also, you can mount the work directory into the container to build the latest documentation by:

```bash
docker run -it --rm -p 8000:8000 -v "${PWD}:/mate" mate-docs
```
