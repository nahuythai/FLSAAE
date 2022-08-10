#! /bin/bash

echo "Starting server"
#python3 server.py & 
#sleep 3

for i in `seq 0 2`; do
    echo "Starting client $i"
    python3 client.py &

done


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
