curl -X POST http://localhost:8000/start_profile \
  -d '{"activities": ["CPU", "CUDA"], "record_shapes": true, "profile_memory": true}'
sleep 3

curl -X POST http://localhost:8000/stop_profile
