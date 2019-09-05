#!/bin/bash

currIndex=0
maxIndex=19

while [ $currIndex -lt $maxIndex ]
do
    condor_run -a accounting_group=aei.prod.ml -a request_memory=8000M python evaluation_script.py $currIndex
    ((counter++))
done
    
