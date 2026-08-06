"""
RequirementsSummaryAgent mit AutoGen 4.7.2
Fasst Ergebnisse des Requirements-Teams zusammen
"""

import os
import json
from typing import Dict, Any, List
from collections import Counter

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config.autogen_config import get_autogen_config, get_agent_config


class RequirementsSummaryAgent:
    """
    RequirementsSummaryAgent mit AutoGen 4.7.2
    Fasst Ergebnisse des Requirements-Teams zusammen
    """
    
    def __init__(self):
        """
        Initialisiert den RequirementsSummaryAgent
        """
        self.config = get_agent_config("requirements_summary")
        
        # AutoGen 4.7.2 Model Client erstellen
        self.model_client = OpenAIChatCompletionClient(
            model=self.config["model"],
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            seed=42,
            temperature=self.config["temperature"],
        )
        
        # AutoGen-Agent erstellen
        self.assistant = AssistantAgent(
            name="requirements_summary_assistant",
            system_message="""You are a requirements summary expert. Your task is to generate a summary of requirements.
            
            Summary should include:
            - Total number of requirements
            - Distribution by tag (e.g., Security, Functional, Performance)
            - Distribution by priority (Must, Should, May)
            - Identified conflicts
            - Identified dependencies
            - Quality score (average of all requirements)
            
            Return the summary in JSON format.""",
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
    
    async def generate_summary(self, requirements: List[Dict[str, Any]], kg: Dict[str, Any] = None, mermaid: str = None) -> Dict[str, Any]:
        """
        Generiert Zusammenfassung der Requirements
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
            kg (Dict[str, Any]): Knowledge Graph (optional)
            mermaid (str): Mermaid-Diagramm (optional)
        
        Returns:
            Dict[str, Any]: Zusammenfassung
        """
        # Statistiken berechnen
        total_requirements = len(requirements)
        by_tag = self._calculate_by_tag(requirements)
        by_priority = self._calculate_by_priority(requirements)
        
        # Konflikte und Abhängigkeiten aus Knowledge Graph extrahieren
        conflicts = []
        dependencies = []
        if kg:
            conflicts = self._extract_conflicts(kg)
            dependencies = self._extract_dependencies(kg)
        
        # Quality-Score berechnen
        quality_score = self._calculate_quality_score(requirements)
        
        # Zusammenfassung für den Prompt vorbereiten
        summary_data = {
            "total_requirements": total_requirements,
            "by_tag": by_tag,
            "by_priority": by_priority,
            "conflicts": conflicts,
            "dependencies": dependencies,
            "quality_score": quality_score
        }
        
        summary_text = json.dumps(summary_data, indent=2)
        
        prompt = f"""Generate a summary of the following requirements:
        
        {summary_text}
        
        Return the summary in JSON format."""
        
        cancellation_token = CancellationToken()
        response = await self.assistant.on_messages(
            [TextMessage(content=prompt, source="user")],
            cancellation_token
        )
        
        # Generierte Zusammenfassung aus der Antwort parsen
        summary = self._parse_summary(response.chat_message.content)
        
        return summary
    
    def _calculate_by_tag(self, requirements: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Berechnet die Verteilung nach Tags
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
        
        Returns:
            Dict[str, int]: Verteilung nach Tags
        """
        tags = [req.get("tag", "Unknown") for req in requirements]
        tag_counts = Counter(tags)
        
        return dict(tag_counts)
    
    def _calculate_by_priority(self, requirements: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Berechnet die Verteilung nach Prioritäten
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
        
        Returns:
            Dict[str, int]: Verteilung nach Prioritäten
        """
        priorities = [req.get("priority", "Unknown") for req in requirements]
        priority_counts = Counter(priorities)
        
        return dict(priority_counts)
    
    def _extract_conflicts(self, kg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrahiert Konflikte aus Knowledge Graph
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            List[Dict[str, Any]]: Liste der Konflikte
        """
        conflicts = []
        
        for edge in kg.get("edges", []):
            if edge.get("type") == "CONFLICTS_WITH":
                conflicts.append(edge)
        
        return conflicts
    
    def _extract_dependencies(self, kg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrahiert Abhängigkeiten aus Knowledge Graph
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            List[Dict[str, Any]]: Liste der Abhängigkeiten
        """
        dependencies = []
        
        for edge in kg.get("edges", []):
            if edge.get("type") == "DEPENDS_ON":
                dependencies.append(edge)
        
        return dependencies
    
    def _calculate_quality_score(self, requirements: List[Dict[str, Any]]) -> float:
        """
        Berechnet den Quality-Score
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
        
        Returns:
            float: Quality-Score (0-100)
        """
        if not requirements:
            return 0.0
        
        # Quality-Score basierend auf Validierung
        valid_requirements = [req for req in requirements if req.get("is_valid", True)]
        
        if not valid_requirements:
            return 0.0
        
        # Quality-Score berechnen (einfache Implementierung)
        quality_score = (len(valid_requirements) / len(requirements)) * 100
        
        return round(quality_score, 2)
    
    def _parse_summary(self, response: str) -> Dict[str, Any]:
        """
        Parst die Zusammenfassung aus der LLM-Antwort
        
        Args:
            response (str): LLM-Antwort
        
        Returns:
            Dict[str, Any]: Zusammenfassung
        """
        try:
            # Versuchen, JSON aus der Antwort zu extrahieren
            if "generate_summary" in response:
                # Function-Calling-Antwort parsen
                import re
                match = re.search(r'"total_requirements":\s*(\d+)', response, re.DOTALL)
                if match:
                    total_requirements = int(match.group(1))
                else:
                    total_requirements = 0
                
                match = re.search(r'"by_tag":\s*\{([^}]*)\}', response, re.DOTALL)
                if match:
                    by_tag = json.loads(f"{{{match.group(1)}}}")
                else:
                    by_tag = {}
                
                match = re.search(r'"by_priority":\s*\{([^}]*)\}', response, re.DOTALL)
                if match:
                    by_priority = json.loads(f"{{{match.group(1)}}}")
                else:
                    by_priority = {}
                
                match = re.search(r'"quality_score":\s*(\d+\.?\d*)', response, re.DOTALL)
                if match:
                    quality_score = float(match.group(1))
                else:
                    quality_score = 0.0
                
                return {
                    "total_requirements": total_requirements,
                    "by_tag": by_tag,
                    "by_priority": by_priority,
                    "conflicts": [],
                    "dependencies": [],
                    "quality_score": quality_score
                }
            
            # Fallback: Versuchen, JSON direkt zu parsen
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Leere Zusammenfassung zurückgeben
            return {
                "total_requirements": 0,
                "by_tag": {},
                "by_priority": {},
                "conflicts": [],
                "dependencies": [],
                "quality_score": 0.0
            }
