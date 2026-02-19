# knowledge_base.py
import json
import os
from typing import List, Dict, Any

class KnowledgeBase:
    def __init__(self):
        self.data_file = 'smartmulch_knowledge.json'
        self.knowledge = self.load_knowledge()
    
    def load_knowledge(self):
        """Load knowledge from JSON file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default knowledge base
            return self.create_default_knowledge()
    
    def create_default_knowledge(self):
        """Create default knowledge base about SmartMulch"""
        knowledge = {
            "company_info": {
                "name": "SmartMulch",
                "tagline": "Intelligent Farming Solutions",
                "founded": "2023",
                "mission": "To revolutionize agriculture through smart technology and sustainable practices",
                "vision": "Making farming smarter, more efficient, and environmentally friendly"
            },
            "products": [
                {
                    "name": "Smart Mulch Film",
                    "description": "Intelligent biodegradable mulch film with moisture sensors",
                    "features": [
                        "Biodegradable material",
                        "Built-in moisture sensors",
                        "UV protected",
                        "Lasts 6-12 months",
                        "Smartphone monitoring"
                    ],
                    "benefits": [
                        "Reduces water usage by 40%",
                        "Prevents weed growth",
                        "Maintains soil temperature",
                        "Improves crop yield"
                    ]
                },
                {
                    "name": "Soil Health Monitor",
                    "description": "IoT device for real-time soil monitoring",
                    "features": [
                        "pH monitoring",
                        "Nutrient level tracking",
                        "Moisture sensing",
                        "Temperature monitoring",
                        "Cloud connectivity"
                    ]
                },
                {
                    "name": "Smart Irrigation Controller",
                    "description": "AI-powered irrigation management system",
                    "features": [
                        "Weather-based scheduling",
                        "Remote control via app",
                        "Water usage analytics",
                        "Leak detection"
                    ]
                }
            ],
            "services": [
                {
                    "name": "Farm Consulting",
                    "description": "Expert advice from agricultural specialists",
                    "details": "One-on-one consultation with agronomists"
                },
                {
                    "name": "Data Analytics",
                    "description": "Advanced analytics for farm optimization",
                    "details": "Yield prediction, resource optimization, market trends"
                },
                {
                    "name": "Training Programs",
                    "description": "Workshops and training for farmers",
                    "details": "Smart farming techniques, equipment training"
                }
            ],
            "farming_guides": {
                "crop_recommendations": {
                    "vegetables": [
                        {"name": "Tomatoes", "season": "Spring-Summer", "water_needs": "Medium", "soil_ph": "6.0-6.8"},
                        {"name": "Peppers", "season": "Spring-Summer", "water_needs": "Medium", "soil_ph": "6.0-6.8"},
                        {"name": "Lettuce", "season": "Spring-Fall", "water_needs": "High", "soil_ph": "6.0-7.0"},
                        {"name": "Carrots", "season": "Spring-Fall", "water_needs": "Medium", "soil_ph": "6.0-6.8"},
                        {"name": "Cucumbers", "season": "Summer", "water_needs": "High", "soil_ph": "6.0-7.0"}
                    ],
                    "fruits": [
                        {"name": "Strawberries", "season": "Spring", "water_needs": "Medium", "soil_ph": "5.5-6.5"},
                        {"name": "Blueberries", "season": "Spring", "water_needs": "Medium", "soil_ph": "4.5-5.5"},
                        {"name": "Apples", "season": "Year-round", "water_needs": "Medium", "soil_ph": "6.0-7.0"}
                    ],
                    "grains": [
                        {"name": "Corn", "season": "Summer", "water_needs": "High", "soil_ph": "6.0-6.8"},
                        {"name": "Wheat", "season": "Winter", "water_needs": "Medium", "soil_ph": "6.0-7.0"}
                    ]
                },
                "soil_types": {
                    "clay": {
                        "characteristics": "Heavy, nutrient-rich, poor drainage",
                        "best_for": ["Rice", "Wheat", "Sugarcane"],
                        "improvement": "Add organic matter, sand, gypsum"
                    },
                    "sandy": {
                        "characteristics": "Light, good drainage, low nutrients",
                        "best_for": ["Carrots", "Potatoes", "Melons"],
                        "improvement": "Add compost, mulch, cover crops"
                    },
                    "loamy": {
                        "characteristics": "Ideal, balanced texture and nutrients",
                        "best_for": ["Most vegetables", "Fruits", "Flowers"],
                        "improvement": "Maintain with organic matter"
                    }
                },
                "irrigation_tips": [
                    "Water early morning to reduce evaporation",
                    "Use drip irrigation for water efficiency",
                    "Mulch to retain soil moisture",
                    "Check soil moisture before watering",
                    "Adjust watering based on weather"
                ],
                "organic_farming": {
                    "composting": "Use kitchen scraps, leaves, grass clippings",
                    "natural_pesticides": ["Neem oil", "Garlic spray", "Soap spray"],
                    "crop_rotation": "Rotate crops yearly to prevent diseases",
                    "companion_planting": [
                        {"plants": "Tomatoes & Basil", "benefit": "Improves growth and flavor"},
                        {"plants": "Corn & Beans", "benefit": "Beans fix nitrogen for corn"},
                        {"plants": "Carrots & Onions", "benefit": "Onions repel carrot flies"}
                    ]
                }
            },
            "faq": [
                {
                    "question": "What is smart mulching?",
                    "answer": "Smart mulching combines traditional mulching techniques with IoT sensors and AI to monitor soil conditions, optimize water usage, and improve crop yields. Our Smart Mulch Film includes moisture sensors that send real-time data to your phone."
                },
                {
                    "question": "How much water can I save?",
                    "answer": "Our smart irrigation systems can reduce water usage by 30-50% compared to traditional methods, depending on your crops and climate."
                },
                {
                    "question": "Is SmartMulch suitable for small farms?",
                    "answer": "Absolutely! We offer solutions for farms of all sizes, from home gardens to large commercial operations. Our products are scalable and cost-effective."
                },
                {
                    "question": "Do I need technical expertise?",
                    "answer": "No, our systems are designed to be user-friendly. The mobile app provides simple instructions and alerts. We also offer training and support."
                }
            ],
            "pricing_info": {
                "smart_mulch_film": "$X per roll",
                "soil_monitor": "$Y per unit",
                "irrigation_controller": "$Z per system",
                "consulting": "Contact for custom quotes",
                "subscription": "Free basic app, premium features available"
            },
            "contact_info": {
                "email": "info@smartmulch.com",
                "phone": "+1 (555) 123-4567",
                "address": "123 Farm Tech Drive, Agriville, AG 12345",
                "hours": "Monday-Friday: 9am-6pm EST"
            }
        }
        
        # Save to file
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False)
        
        return knowledge
    
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information"""
        results = []
        query_lower = query.lower()
        
        # Search through all sections
        def search_dict(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if query_lower in str(key).lower() or query_lower in str(value).lower():
                        results.append({
                            "path": new_path,
                            "content": value
                        })
                    search_dict(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_dict(item, f"{path}[{i}]")
        
        search_dict(self.knowledge)
        return results[:10]  # Return top 10 results
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for AI prompt"""
        relevant_info = self.search_knowledge(query)
        
        if not relevant_info:
            return "No specific information found. Provide general helpful response."
        
        context = "Here's relevant information from SmartMulch knowledge base:\n\n"
        for item in relevant_info:
            if isinstance(item['content'], (dict, list)):
                context += f"- {item['path']}: {json.dumps(item['content'], indent=2)[:200]}...\n"
            else:
                context += f"- {item['path']}: {item['content']}\n"
        
        return context

# Create singleton instance
knowledge_base = KnowledgeBase()
