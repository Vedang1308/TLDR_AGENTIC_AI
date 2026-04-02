# Copyright Sierra

import json
from typing import Any, Dict, List
from tau_bench.envs.tool import Tool

class UpdateReservationPayment(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], reservation_id: str, payment_methods: List[Dict[str, Any]]) -> str:
        reserations = data["reservations"]
        if reservation_id in reserations:
            reserations[reservation_id]["payment_history"] = payment_methods
            return json.dumps(reserations[reservation_id])
        return f"Error: reservation {reservation_id} not found"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "update_reservation_payment",
                "description": "Update the payment method of a reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation id, such as 'HATHAT'.",
                        },
                        "payment_methods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "payment_id": {
                                        "type": "string",
                                        "description": "The payment id, such as 'credit_card_7815826'.",
                                    },
                                    "amount": {
                                        "type": "number",
                                        "description": "The amount to be paid.",
                                    },
                                },
                                "required": ["payment_id", "amount"],
                            },
                        },
                    },
                    "required": ["reservation_id", "payment_methods"],
                },
            },
        }
