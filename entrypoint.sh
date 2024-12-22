#!/bin/bash

JN_PORT=${PORT:-8888}
if [ -z "$JN_IP" ]; then
    JN_IP=$(hostname -I | awk '{print $1}')
fi

jupyter notebook --allow-root --ip="$JN_IP" --port="$JN_PORT" --no-browser
