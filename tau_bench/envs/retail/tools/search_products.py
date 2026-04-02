# Copyright Sierra

import json
from typing import Any, Dict, List
from tau_bench.envs.tool import Tool

class SearchProducts(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], query: str) -> str:
        products = data["products"]
        results = []
        query_lower = query.lower()
        
        for p_id, product in products.items():
            # Search in name
            if query_lower in product["name"].lower():
                results.append(product)
                continue
                
            # Search in variants/options if needed, but official search usually hits name/category
            # For robustness, we check the variant options too
            match_found = False
            for variant in product.get("variants", {}).values():
                for opt_val in variant.get("options", {}).values():
                    if query_lower in str(opt_val).lower():
                        results.append(product)
                        match_found = True
                        break
                if match_found:
                    break
                    
        return json.dumps(results[:10]) # Limit to top 10 results for context safety

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search for products in the catalog by name or attributes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, e.g. 'Laptop' or 'blue shirt'.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
