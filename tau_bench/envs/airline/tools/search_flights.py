# Copyright Sierra

import json
from typing import Any, Dict, List
from tau_bench.envs.tool import Tool


class SearchFlights(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], origin: str, destination: str, date: str) -> str:
        flights = data["flights"]
        results = []
        
        # 1. Direct Flights
        for f_id, flight in flights.items():
            if flight["origin"] == origin and flight["destination"] == destination:
                if date in flight["dates"] and flight["dates"][date]["status"] == "available":
                    res = {
                        "flight_number": f_id,
                        "type": "direct",
                        "origin": origin,
                        "destination": destination,
                        "status": "available"
                    }
                    results.append(res)
        
        # 2. 1-Stopover Flights (Compositional)
        # Find A -> C and C -> B on the same date
        origins = [f for f in flights.values() if f["origin"] == origin]
        destinations = [f for f in flights.values() if f["destination"] == destination]
        
        for f1 in origins:
            for f2 in destinations:
                if f1["destination"] == f2["origin"] and f1["flight_number"] != f2["flight_number"]:
                    # Found a path A -> C -> B
                    if date in f1["dates"] and f1["dates"][date]["status"] == "available":
                        if date in f2["dates"] and f2["dates"][date]["status"] == "available":
                            results.append({
                                "type": "onestop",
                                "connection": f1["destination"],
                                "flights": [f1["flight_number"], f2["flight_number"]],
                                "status": "available"
                            })
                            
        return json.dumps(results)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Find available flights between two cities (direct or 1-stop) on a specific date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "The IATA origin code, such as 'PHL'.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The IATA destination code, such as 'SEA'.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The flight date in YYYY-MM-DD format.",
                        },
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        }
