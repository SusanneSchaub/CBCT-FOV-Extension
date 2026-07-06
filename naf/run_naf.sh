#!/bin/bash

numbers=(98)  # 8.5. 12:19 next time run this!

for num in "${numbers[@]}"
do
    echo "Running with $num..."
    python3 train_adapt.py --num "$num"

    # wait for script to finish before continuing (default behavior)
    echo "Finished $num"
done

echo "All runs completed."
