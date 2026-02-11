
#Generating Traces from vLLM
#Add to your vLLM startup logging:
# In your vLLM request handler
import json
import time

log_file = open("trace.json", "w")
log_entries = []

def log_request(prompt, completion):
    entry = {
        "timestamp": time.time() - start_time,
        "input_tokens": len(prompt) // 4,  # Approximate
        "output_tokens": len(completion) // 4
    }
    log_entries.append(entry)
    
    # Periodic flush
    if len(log_entries) % 100 == 0:
        json.dump(log_entries, log_file)
        log_file.flush()
