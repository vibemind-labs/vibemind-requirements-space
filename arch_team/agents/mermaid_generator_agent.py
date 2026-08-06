"""
MermaidGeneratorAgent mit AutoGen 4.7.2
Generiert automatisch Mermaid-Diagramm aus Knowledge Graph
"""

import os
import json
from typing import Dict, Any, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config.autogen_config import get_autogen_config, get_agent_config, get_diagram_types


class MermaidGeneratorAgent:
    """
    MermaidGeneratorAgent mit AutoGen 4.7.2
    Generiert automatisch Mermaid-Diagramm aus Knowledge Graph
    """
    
    def __init__(self):
        """
        Initialisiert den MermaidGeneratorAgent
        """
        self.config = get_agent_config("mermaid_generator")
        
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
            name="mermaid_generator_assistant",
            system_message="""You are a Mermaid diagram expert. Your task is to generate Mermaid diagrams from knowledge graphs.
            
            Diagram types:
            - dependency: Shows dependencies between requirements
            - conflict: Shows conflicts between requirements
            - grouped: Shows requirements grouped by tags
            - actor: Shows actor-requirement relationships
            
            Mermaid syntax:
            - graph TD: Top-down graph
            - graph LR: Left-right graph
            - subgraph: Grouped nodes
            - -->: Arrow edge
            - -.x.: Dotted edge for conflicts
            
            Return Mermaid diagram in text format.""",
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
    
    async def generate_mermaid_diagram(self, kg: Dict[str, Any], diagram_type: str = "dependency") -> Dict[str, Any]:
        """
        Generiert Mermaid-Diagramm aus Knowledge Graph
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph mit Nodes und Edges
            diagram_type (str): Diagramm-Typ (dependency, conflict, grouped, actor)
        
        Returns:
            Dict[str, Any]: Mermaid-Diagramm mit Metadaten
        """
        # Knowledge Graph für den Prompt vorbereiten
        kg_text = json.dumps(kg, indent=2)
        
        prompt = f"""Generate a {diagram_type} Mermaid diagram from following knowledge graph:
        
        {kg_text}
        
        Available diagram types: {', '.join(get_diagram_types())}
        
        Return Mermaid diagram in text format with metadata."""
        
        cancellation_token = CancellationToken()
        response = await self.assistant.on_messages(
            [TextMessage(content=prompt, source="user")],
            cancellation_token
        )
        
        # Generiertes Mermaid-Diagramm aus der Antwort parsen
        mermaid_result = self._parse_mermaid_diagram(response.chat_message.content)
        
        return mermaid_result
    
    async def validate_mermaid_syntax(self, mermaid: str) -> Dict[str, Any]:
        """
        Validiert die Mermaid-Syntax
        
        Args:
            mermaid (str): Mermaid-Diagramm
        
        Returns:
            Dict[str, Any]: Validierungsergebnis
        """
        # Einfache Syntax-Validierung
        is_valid = True
        issues = []
        
        # Prüfen, ob das Diagramm mit einem gültigen Typ beginnt
        valid_starts = ["graph TD", "graph LR", "flowchart TD", "flowchart LR"]
        if not any(mermaid.strip().startswith(start) for start in valid_starts):
            is_valid = False
            issues.append("Diagram does not start with a valid type (graph TD, graph LR, flowchart TD, flowchart LR)")
        
        # Prüfen, ob das Diagramm mit einem gültigen Ende endet
        if not mermaid.strip().endswith("}"):
            is_valid = False
            issues.append("Diagram does not end with }")
        
        # Prüfen, ob das Diagramm Klammern hat
        if mermaid.count("{") != mermaid.count("}"):
            is_valid = False
            issues.append("Unbalanced braces")
        
        return {
            "valid": is_valid,
            "issues": issues
        }
    
    def _parse_mermaid_diagram(self, response: str) -> Dict[str, Any]:
        """
        Parst das Mermaid-Diagramm aus der LLM-Antwort
        
        Args:
            response (str): LLM-Antwort
        
        Returns:
            Dict[str, Any]: Mermaid-Diagramm mit Metadaten
        """
        try:
            # Versuchen, JSON aus der Antwort zu extrahieren
            if "generate_mermaid_diagram" in response:
                # Function-Calling-Antwort parsen
                import re
                match = re.search(r'"mermaid":\s*"([^"]*)"', response, re.DOTALL)
                if match:
                    mermaid = match.group(1)
                else:
                    mermaid = ""
                
                match = re.search(r'"diagram_type":\s*"([^"]*)"', response, re.DOTALL)
                if match:
                    diagram_type = match.group(1)
                else:
                    diagram_type = "dependency"
                
                match = re.search(r'"metadata":\s*\{([^}]*)\}', response, re.DOTALL)
                if match:
                    metadata = json.loads(f"{{{match.group(1)}}}")
                else:
                    metadata = {}
                
                return {
                    "mermaid": mermaid,
                    "diagram_type": diagram_type,
                    "metadata": metadata
                }
            
            # Fallback: Versuchen, JSON direkt zu parsen
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Leeres Mermaid-Diagramm zurückgeben
            return {
                "mermaid": "",
                "diagram_type": "dependency",
                "metadata": {}
            }
    
    async def generate_dependency_diagram(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert ein Dependency-Diagramm
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            Dict[str, Any]: Mermaid-Diagramm
        """
        return await self.generate_mermaid_diagram(kg, "dependency")
    
    async def generate_conflict_diagram(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert ein Conflict-Diagramm
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            Dict[str, Any]: Mermaid-Diagramm
        """
        return await self.generate_mermaid_diagram(kg, "conflict")
    
    async def generate_grouped_diagram(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert ein Grouped-Diagramm
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            Dict[str, Any]: Mermaid-Diagramm
        """
        return await self.generate_mermaid_diagram(kg, "grouped")
    
    async def generate_actor_diagram(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert ein Actor-Diagramm
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            Dict[str, Any]: Mermaid-Diagramm
        """
        return await self.generate_mermaid_diagram(kg, "actor")
