#!/bin/bash

N=9
for i in $(seq 0 $N); do
    scp julian.bopp@dbe-cia-sl19-01.dbe.unibas.ch:/home/julian.bopp/samples/sample_"$i".nii.gz /Users/pewiby59/Thesis/cbct-artifact-reduction/
done
