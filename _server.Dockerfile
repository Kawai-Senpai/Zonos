FROM zonos-base:latest

WORKDIR /app

COPY assets/ assets/
COPY zonos/ zonos/

COPY config.py ./
COPY sample.py ./
COPY menu_synthesize.py ./

EXPOSE 7860
