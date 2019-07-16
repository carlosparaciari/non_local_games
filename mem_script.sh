#!/bin/bash

FILEPID="./process_PID.txt"
FILEDAT="./mem_usage.dat"
PID=$(cat "$FILEPID")

while true; do OUTPUT="$(ps -p ${PID} -o %mem | sed -n 2p)"; echo "${OUTPUT}" >> "$FILEDAT"; sleep 2; done
