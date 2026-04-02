# Copyright Sierra

import json
from typing import Any, Dict
from tau_bench.envs.tool import Tool

class GetFlightDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], flight_number: str) -> str:
        flights = data["flights"]
        if flight_number in flights:
            return json.dumps(flights[flight_number])
        return f"Error: flight {flight_number} not found"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_flight_details",
                "description": "Get departure/arrival times, aircraft info, and class pricing for a flight.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_number": {
                            "type": "string",
                            "description": "The flight number, such as 'HAT001'.",
                        },
                    },
                    "required": ["flight_number"],
                },
            },
        }
