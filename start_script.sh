#!/bin/bash

nohup python3 main_SDP_n1_2_n2_2.py > output.txt 2>&1 &
echo $! > process_PID.txt

sleep 2

nohup ./mem_script.sh > mem.log 2>&1 &
echo $! > mem_PID.txt
