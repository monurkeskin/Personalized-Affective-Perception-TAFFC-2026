"""Inert transport diagnostic; never used as a configured policy fallback."""
import json
import sys


def main():
    for line in sys.stdin:
        packet = json.loads(line)
        operation = packet["operation"]
        if operation == "hello":
            result = {"runtime": "synthetic-transport", "capabilities": ["present"],
                      "hardware_tested": False}
        elif operation == "present":
            result = {"delivered": True, "synthetic": True, **packet["payload"]}
        elif operation == "close":
            return
        else:
            raise ValueError("Unknown synthetic operation.")
        reply = {key: packet[key] for key in ("protocol", "session_id", "request_id")}
        reply.update(ok=True, result=result)
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
