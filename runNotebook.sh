#!/usr/bin/env bash
exec docker run --rm -it --name mlnd-analysis-notebook -p 8888:8888 -v $PWD:/home/jovyan/work analysis-notebook

