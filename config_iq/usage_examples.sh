# 1. Get model info
python roofline-recommend.py model-info --model-id meta-llama/Llama-2-7b-hf

# 2. Get workload defaults for chatbot
python roofline-recommend.py workload-defaults --deployment-type chatbot

# 3. List hardware (all AWS A10G instances under $5/hr)
python roofline-recommend.py hardware-ls --cloud aws --gpu-type A10G --max-price 5

# 4. Estimate performance for a single configuration
python roofline-recommend.py roofline-estimate \
    --model-id meta-llama/Llama-2-7b-hf \
    --deployment-type chatbot \
    --hardware-id aws:g5.xlarge

# 5. Search for optimal configurations (ranks by cost, applies headroom)
python roofline-recommend.py search-configs \
    --model-id meta-llama/Llama-2-7b-hf \
    --deployment-type chatbot \
    --safety-margin 0.3 \
    --max-headroom 0.6
