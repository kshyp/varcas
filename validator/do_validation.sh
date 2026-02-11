python varcas_validator.py compare \
    --baseline baseline.json \
    --candidate optimized.json \
    --epsilon-abs 0.02 \
    --epsilon-rel 5.0 \
    --output degradation_report.json
