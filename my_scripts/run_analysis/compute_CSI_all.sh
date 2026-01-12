cd ..

# Define upstream and downstream datasets
upstream=("ani1x" "transition1x" "oc20" "oc22")
downstream=("rmd17" "md22" "qm9" "spice" "qmof" "matbench_fold0")

# Loop through upstream and downstream datasets
for down in "${downstream[@]}"; do
    for up in "${upstream[@]}"; do
        python compute_csi.py \
            "${up}_Node_Seed0_Samplingbalanced.pt" \
            "${down}_Node_Seed0_Samplingrandom.pt" \
            --model_name "equiformer_v2" 
    done
done





