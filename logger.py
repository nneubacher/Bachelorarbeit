from mitmproxy import http
import json

def response(flow: http.HTTPFlow) -> None:
    # Construct the log entry
    log_entry = {
        "request": {
            "url": flow.request.pretty_url,
            "method": flow.request.method,
            "headers": dict(flow.request.headers),
            "body": flow.request.text,
        },
        "response": {
            "status_code": flow.response.status_code,
            "headers": dict(flow.response.headers),
            "body": flow.response.text,
        }
    }

    # Append the log entry to a JSON log file
    with open("api_logs.json", "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")  # Newline for readability/separation of entries