import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import requests


class VoiceInterfaceService:
    """
    ElevenLabs voice interface with multi-agent routing
    Agents: parking_assistant, traffic_advisor, payment_coordinator, support_specialist
    """
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.elevenlabs.api_key
        self.voice_id = config.elevenlabs.voice_id
        self.model = config.elevenlabs.model
        
        self.base_url = "https://api.elevenlabs.io/v1"
        
        self.agents = {
            'parking_assistant': {
                'name': 'Parking Assistant',
                'model': config.google_cloud.gemini_model,
                'capabilities': ['spot_finding', 'directions', 'cost_estimation', 'restrictions'],
                'system_prompt': (
                    "You are a helpful parking assistant. Guide users to find available parking spots, "
                    "provide directions, estimate costs, and explain parking restrictions. "
                    "Be concise, friendly, and focus on solving parking needs."
                )
            },
            'traffic_advisor': {
                'name': 'Traffic Advisor',
                'model': config.google_cloud.gemini_model,
                'capabilities': ['traffic_analysis', 'route_optimization', 'congestion_prediction'],
                'system_prompt': (
                    "You are a traffic advisor. Analyze traffic conditions, suggest optimal routes, "
                    "predict congestion patterns, and help users navigate efficiently. "
                    "Provide real-time traffic insights and alternative routes."
                )
            },
            'payment_coordinator': {
                'name': 'Payment Coordinator',
                'model': config.google_cloud.gemini_model,
                'capabilities': ['payment_processing', 'reservations', 'refunds', 'billing'],
                'system_prompt': (
                    "You are a payment coordinator. Handle payment processing, parking reservations, "
                    "refunds, and billing inquiries. Be professional, secure, and clear about costs. "
                    "Explain payment options and confirm transactions."
                )
            },
            'support_specialist': {
                'name': 'Support Specialist',
                'model': config.google_cloud.gemini_model,
                'capabilities': ['issue_resolution', 'technical_support', 'escalation', 'feedback'],
                'system_prompt': (
                    "You are a customer support specialist. Resolve user issues, provide technical support, "
                    "escalate complex problems, and collect feedback. Be empathetic, patient, and solution-oriented."
                )
            }
        }
        
        self.conversation_history = []
        self.current_agent = 'parking_assistant'
    
    def route_conversation(self, user_input: str) -> str:
        """
        Route conversation to appropriate agent based on user input
        Returns agent ID
        """
        user_input_lower = user_input.lower()
        
        parking_keywords = ['find', 'search', 'parking', 'spot', 'available', 'where', 'locate']
        traffic_keywords = ['traffic', 'route', 'congestion', 'fastest', 'avoid', 'highway']
        payment_keywords = ['pay', 'reserve', 'book', 'cost', 'price', 'refund', 'cancel']
        support_keywords = ['help', 'issue', 'problem', 'error', 'complaint', 'support']
        
        if any(keyword in user_input_lower for keyword in parking_keywords):
            return 'parking_assistant'
        elif any(keyword in user_input_lower for keyword in traffic_keywords):
            return 'traffic_advisor'
        elif any(keyword in user_input_lower for keyword in payment_keywords):
            return 'payment_coordinator'
        elif any(keyword in user_input_lower for keyword in support_keywords):
            return 'support_specialist'
        else:
            return self.current_agent
    
    def process_with_gemini(self, user_input: str, agent_id: str) -> str:
        """
        Process user input with Gemini model
        Simulates Gemini API call with agent-specific system prompt
        """
        agent = self.agents[agent_id]
        
        system_prompt = agent['system_prompt']
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]
        ])
        
        full_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_context}\n\nUser: {user_input}\n\nAssistant:"
        
        if agent_id == 'parking_assistant':
            responses = [
                "I found 3 available parking spots within 0.5 miles of your location. The closest one is on Main Street with a rate of $3 per hour. Would you like directions?",
                "Based on current availability, I recommend the garage on 5th Avenue. It has 15 available spots and costs $2.50 per hour. It's about a 3-minute walk to your destination.",
                "There are several spots near you. The most affordable option is street parking on Oak Street at $2 per hour, and it's currently showing 70% availability."
            ]
        elif agent_id == 'traffic_advisor':
            responses = [
                "Current traffic on Highway 101 shows moderate congestion. I recommend taking the alternate route via Market Street, which will save you approximately 8 minutes.",
                "Traffic conditions are light right now. The fastest route to your destination is via the downtown corridor, estimated at 12 minutes.",
                "There's heavy traffic on your usual route due to an accident. I suggest taking the side streets through the residential area, adding only 3 minutes to your trip."
            ]
        elif agent_id == 'payment_coordinator':
            responses = [
                "Your reservation has been confirmed. The total cost for 2 hours is $6.00. I've sent a confirmation email with your parking details and receipt.",
                "I can process your payment now. The parking rate is $4 per hour, and you've indicated a 3-hour stay, totaling $12. Would you like to proceed?",
                "Your refund request has been approved. The $8.00 will be credited back to your original payment method within 3-5 business days."
            ]
        else:
            responses = [
                "I understand you're experiencing an issue with the parking meter. Let me help you troubleshoot this. Can you tell me the meter number?",
                "Thank you for your feedback. I've logged your suggestion about adding more handicapped-accessible spots. Our team will review this for future improvements.",
                "I apologize for the inconvenience. I'm escalating your issue to our technical team. You should receive a response within 24 hours."
            ]
        
        import random
        response = random.choice(responses)
        
        return response
    
    def synthesize_speech(self, text: str) -> Dict:
        """
        Convert text to speech using ElevenLabs API
        Returns audio data and metadata
        """
        if not self.api_key or not self.voice_id:
            print("ElevenLabs API credentials not configured")
            return {
                'success': False,
                'audio_length_chars': len(text),
                'estimated_duration_seconds': len(text) / 15
            }
        
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.config.elevenlabs.stability,
                "similarity_boost": self.config.elevenlabs.similarity_boost,
                "style": self.config.elevenlabs.style
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                audio_data = response.content
                
                return {
                    'success': True,
                    'audio_data': audio_data,
                    'audio_length_bytes': len(audio_data),
                    'text_length_chars': len(text),
                    'estimated_duration_seconds': len(text) / 15
                }
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}",
                    'text_length_chars': len(text)
                }
        
        except Exception as e:
            print(f"Speech synthesis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'text_length_chars': len(text)
            }
    
    def handle_voice_query(self, user_input: str, synthesize: bool = False) -> Dict:
        """
        Handle complete voice query with agent routing and speech synthesis
        Returns response text, agent info, and audio data
        """
        start_time = time.time()
        
        agent_id = self.route_conversation(user_input)
        self.current_agent = agent_id
        
        response_text = self.process_with_gemini(user_input, agent_id)
        
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response_text,
            'agent': agent_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        audio_data = None
        if synthesize:
            audio_result = self.synthesize_speech(response_text)
            audio_data = audio_result
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            'user_input': user_input,
            'agent_id': agent_id,
            'agent_name': self.agents[agent_id]['name'],
            'response_text': response_text,
            'response_length_chars': len(response_text),
            'processing_time_ms': processing_time,
            'audio_data': audio_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return result
    
    def simulate_voice_conversations(self, num_queries: int = 10) -> List[Dict]:
        """
        Simulate voice conversations with various query types
        Returns conversation metrics and agent distribution
        """
        queries = [
            "Find me a parking spot near 123 Main Street",
            "What's the traffic like on Highway 101 right now?",
            "I need to pay for my parking reservation",
            "I'm having trouble with the parking meter",
            "Where can I park for free downtown?",
            "What's the fastest route to avoid traffic?",
            "Can I get a refund for my unused parking time?",
            "How do I report a broken parking meter?",
            "Show me available spots near the shopping mall",
            "Is there heavy traffic on Market Street?"
        ]
        
        results = []
        agent_usage = {'parking_assistant': 0, 'traffic_advisor': 0, 
                      'payment_coordinator': 0, 'support_specialist': 0}
        
        for i in range(min(num_queries, len(queries))):
            query = queries[i]
            result = self.handle_voice_query(query, synthesize=False)
            results.append(result)
            agent_usage[result['agent_id']] += 1
            
            print(f"Query {i+1}: {query[:50]}... -> Agent: {result['agent_name']}")
        
        avg_response_length = sum(r['response_length_chars'] for r in results) / len(results)
        
        summary = {
            'total_queries': len(results),
            'agent_distribution': agent_usage,
            'average_response_length_chars': avg_response_length,
            'conversations': results
        }
        
        return summary
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.current_agent = 'parking_assistant'
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict]:
        """Get information about specific agent"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> Dict:
        """Get information about all agents"""
        return {
            agent_id: {
                'name': agent['name'],
                'capabilities': agent['capabilities']
            }
            for agent_id, agent in self.agents.items()
        }