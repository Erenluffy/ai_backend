# knowledge_base.py
import json
import os
from typing import List, Dict, Any, Optional
import re

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
            print(f"Warning: {self.data_file} not found. Using default knowledge.")
            return self.create_default_knowledge()
    
    def create_default_knowledge(self):
        """Create default knowledge base if file doesn't exist"""
        # This should match the JSON structure above
        return {
            "company_info": {
                "name": "SmartMulch",
                "tagline": "Intelligent Farming Solutions"
            },
            "message": "Please ensure smartmulch_knowledge.json is properly configured with your research data."
        }
    
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Intelligent search through knowledge base"""
        results = []
        query_lower = query.lower()
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        def search_dict(obj, path="", score=0):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Calculate relevance score
                    item_score = self._calculate_relevance(key, value, keywords, query_lower)
                    
                    if item_score > 0:
                        results.append({
                            "path": new_path,
                            "content": value,
                            "score": item_score,
                            "key": key
                        })
                    
                    # Continue searching deeper
                    search_dict(value, new_path, score + item_score)
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_dict(item, f"{path}[{i}]", score)
        
        search_dict(self.knowledge)
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:15]  # Return top 15 results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        common_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'how', 'what', 
                       'when', 'where', 'why', 'which', 'who', 'whom', 'this', 'that',
                       'these', 'those', 'for', 'to', 'with', 'about', 'can', 'you',
                       'your', 'tell', 'me', 'know', 'please', 'thanks', 'thank']
        
        words = query.split()
        keywords = [w for w in words if w not in common_words and len(w) > 2]
        
        # Add specific domain terms
        domain_terms = {
            'moisture': ['moisture', 'water', 'wet', 'dry', 'humid'],
            'nutrient': ['nutrient', 'nitrogen', 'phosphorus', 'potassium', 'npk', 'n', 'p', 'k', 'fertilizer'],
            'temperature': ['temperature', 'temp', 'hot', 'cold', 'warm', 'cool'],
            'ph': ['ph', 'acidic', 'alkaline', 'acidity'],
            'ec': ['ec', 'electrical', 'conductivity', 'salt', 'salinity'],
            'mulch': ['mulch', 'mulching', 'biodegradable', 'plastic'],
            'research': ['research', 'study', 'experiment', 'finding', 'result', 'data'],
            'sensor': ['sensor', 'iot', 'monitor', 'measure', 'device'],
            'crop': ['crop', 'tomato', 'vegetable', 'fruit', 'plant', 'grow'],
            'soil': ['soil', 'earth', 'ground', 'dirt'],
            'benefit': ['benefit', 'advantage', 'improve', 'better', 'increase']
        }
        
        # Add related terms
        expanded_keywords = set(keywords)
        for word in keywords:
            for term, synonyms in domain_terms.items():
                if word in synonyms or any(s in word for s in synonyms):
                    expanded_keywords.add(term)
                    expanded_keywords.update(synonyms)
        
        return list(expanded_keywords)
    
    def _calculate_relevance(self, key: str, value: any, keywords: List[str], query: str) -> int:
        """Calculate relevance score for a piece of information"""
        score = 0
        text = f"{key} {str(value)}".lower()
        
        # Direct keyword matches
        for keyword in keywords:
            if keyword in text:
                # Weight by keyword importance
                if keyword in ['moisture', 'water']:
                    score += 10
                elif keyword in ['nitrogen', 'phosphorus', 'potassium', 'npk']:
                    score += 9
                elif keyword in ['temperature', 'ph', 'ec']:
                    score += 8
                elif keyword in ['mulch', 'biodegradable']:
                    score += 7
                elif keyword in ['research', 'study', 'experiment']:
                    score += 6
                elif keyword in ['sensor', 'iot']:
                    score += 5
                else:
                    score += 3
        
        # Boost for numbers/percentages (research data)
        if re.search(r'\d+\.?\d*%?', text):
            score += 4
            
        # Boost for comparative terms
        if any(term in text for term in ['vs', 'versus', 'compared', 'better', 'worse']):
            score += 3
            
        # Boost for table data
        if 'table' in text or re.search(r'\d+\.\d+\s*vs\s*\d+\.\d+', text):
            score += 5
            
        return score
    
    def get_context_for_query(self, query: str) -> str:
        """Get rich context for AI prompt"""
        relevant_info = self.search_knowledge(query)
        
        if not relevant_info:
            return "I don't have specific information about that in my knowledge base, but I'll do my best to help based on general agricultural knowledge."
        
        # Group by category
        categorized = {}
        for item in relevant_info:
            category = item['path'].split('.')[0] if '.' in item['path'] else 'general'
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        
        context = "Here's relevant information from the SmartMulch knowledge base and research data:\n\n"
        
        # Priority order for categories
        priority = ['research_findings', 'faq', 'products', 'comparative_tables', 
                   'farming_guides', 'experimental_setup', 'technical_details']
        
        # Add prioritized categories first
        for cat in priority:
            if cat in categorized:
                context += f"\n📊 {cat.replace('_', ' ').title()}:\n"
                for item in categorized[cat][:3]:  # Limit to 3 per category
                    if isinstance(item['content'], (dict, list)):
                        if 'question' in str(item['content']) and 'answer' in str(item['content']):
                            # Format FAQ nicely
                            if isinstance(item['content'], dict) and 'question' in item['content']:
                                context += f"  • Q: {item['content']['question']}\n    A: {item['content']['answer'][:200]}...\n"
                            else:
                                context += f"  • {item['path'].split('.')[-1]}: {json.dumps(item['content'], indent=2)[:200]}...\n"
                        else:
                            context += f"  • {item['path'].split('.')[-1]}: {json.dumps(item['content'], indent=2)[:200]}...\n"
                    else:
                        context += f"  • {item['path'].split('.')[-1]}: {item['content']}\n"
        
        # Add remaining categories
        for cat, items in categorized.items():
            if cat not in priority:
                context += f"\n📌 {cat.replace('_', ' ').title()}:\n"
                for item in items[:2]:  # Limit to 2 per other category
                    if isinstance(item['content'], (dict, list)):
                        context += f"  • {item['path'].split('.')[-1]}: {json.dumps(item['content'], indent=2)[:150]}...\n"
                    else:
                        context += f"  • {item['path'].split('.')[-1]}: {item['content']}\n"
        
        # Add data highlights if relevant
        if any(k in query.lower() for k in ['moisture', 'water', 'nutrient', 'npk', 'compare', 'vs']):
            context += "\n🔬 Key Research Data Highlights:\n"
            
            # Add comparative data
            if 'comparative_tables' in self.knowledge:
                tables = self.knowledge['comparative_tables']
                context += f"  • Moisture: Mulch {tables.get('moisture_analysis', {}).get('mulch', 'N/A')}% vs No Mulch {tables.get('moisture_analysis', {}).get('no_mulch', 'N/A')}% (77% higher with mulch)\n"
                context += f"  • EC: Mulch {tables.get('parameter_wise_comparison', {}).get('EC', {}).get('mulch', 'N/A')} vs No Mulch {tables.get('parameter_wise_comparison', {}).get('EC', {}).get('no_mulch', 'N/A')} (9% lower salts)\n"
                context += f"  • Soil Health Index: Mulch {tables.get('soil_health_index', {}).get('mulch', 'N/A')} vs No Mulch {tables.get('soil_health_index', {}).get('no_mulch', 'N/A')} (14.3% better)\n"
                context += """
        
                IMPORTANT FORMATTING INSTRUCTIONS:
                - NEVER use ** or markdown formatting
                - Use simple bullet points (•) for lists
                - Use numbers (1., 2., etc.) for steps
                - Keep paragraphs short and readable
                - Present data clearly with numbers and percentages
                - Use plain text headings followed by colons
                - Separate sections with blank lines
                """
        return context

# Create singleton instance
knowledge_base = KnowledgeBase()
