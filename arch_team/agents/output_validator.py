"""
OutputValidator mit AutoGen 4.7.2
Validiert den Output auf Syntax, Konsistenz, Vollständigkeit und Semantik
"""

import os
import json
from typing import Dict, Any, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config.autogen_config import get_autogen_config, get_agent_config, get_validation_config


class OutputValidator:
    """
    OutputValidator mit AutoGen 4.7.2
    Validiert den Output auf Syntax, Konsistenz, Vollständigkeit und Semantik
    """
    
    def __init__(self):
        """
        Initialisiert den OutputValidator
        """
        self.config = get_agent_config("output_validator")
        
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
            name="output_validator_assistant",
            system_message="""You are an output validation expert. Your task is to validate the output of the requirements analysis system.
            
            Validation checks:
            - Syntax check: Mermaid diagram is syntactically correct
            - Consistency check: All requirements in KG are in the diagram
            - Completeness check: All relationships are captured
            - Semantic check: Relationships are logically correct (no cycles)
            - Cross-reference check: Chunks are correctly referenced
            
            Return the validation result in JSON format.""",
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
    
    async def validate_syntax(self, mermaid: str) -> Dict[str, Any]:
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
    
    async def validate_consistency(self, kg: Dict[str, Any], mermaid: str) -> Dict[str, Any]:
        """
        Validiert die Konsistenz zwischen Knowledge Graph und Mermaid-Diagramm
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
            mermaid (str): Mermaid-Diagramm
        
        Returns:
            Dict[str, Any]: Validierungsergebnis
        """
        is_valid = True
        issues = []
        
        # Prüfen, ob alle Requirements im Knowledge Graph auch im Mermaid-Diagramm sind
        kg_requirements = [node["id"] for node in kg.get("nodes", []) if node.get("type") == "Requirement"]
        
        for req_id in kg_requirements:
            if req_id not in mermaid:
                is_valid = False
                issues.append(f"Requirement {req_id} is missing from Mermaid diagram")
        
        return {
            "valid": is_valid,
            "issues": issues
        }
    
    async def validate_completeness(self, kg: Dict[str, Any], mermaid: str) -> Dict[str, Any]:
        """
        Validiert die Vollständigkeit
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
            mermaid (str): Mermaid-Diagramm
        
        Returns:
            Dict[str, Any]: Validierungsergebnis
        """
        is_valid = True
        issues = []
        
        # Prüfen, ob alle Beziehungen erfasst sind
        kg_nodes = [node["id"] for node in kg.get("nodes", [])]
        
        for edge in kg.get("edges", []):
            if edge["source"] not in kg_nodes:
                is_valid = False
                issues.append(f"Source node {edge['source']} is missing from KG")
            
            if edge["target"] not in kg_nodes:
                is_valid = False
                issues.append(f"Target node {edge['target']} is missing from KG")
        
        return {
            "valid": is_valid,
            "issues": issues
        }
    
    async def validate_semantic(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validiert die Semantik (keine Zyklen in Abhängigkeiten)
        
        Args:
            kg (Dict[str, Any]): Knowledge Graph
        
        Returns:
            Dict[str, Any]: Validierungsergebnis
        """
        is_valid = True
        issues = []
        
        # Zyklen in Abhängigkeiten erkennen
        dependencies = [edge for edge in kg.get("edges", []) if edge.get("type") == "DEPENDS_ON"]
        
        # Einfache Zyklen-Erkennung
        visited = set()
        for dep in dependencies:
            if dep["target"] in visited:
                is_valid = False
                issues.append(f"Cycle detected: {dep['source']} -> {dep['target']}")
            visited.add(dep["source"])
        
        return {
            "valid": is_valid,
            "issues": issues
        }
    
    async def validate_cross_reference(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validiert die Cross-References
        
        Args:
            requirements (List[Dict[str, Any]]: Liste der Requirements
        
        Returns:
            Dict[str, Any]: Validierungsergebnis
        """
        is_valid = True
        issues = []
        
        # Prüfen, ob alle Evidence-Refs gültig sind
        for req in requirements:
            evidence_refs = req.get("evidence_refs", [])
            
            for ref in evidence_refs:
                if not ref.startswith("chunk-"):
                    is_valid = False
                    issues.append(f"Invalid evidence reference: {ref}")
        
        return {
            "valid": is_valid,
            "issues": issues
        }
    
    async def validate_output(self, mermaid: str, kg: Dict[str, Any] = None, requirements: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validiert den kompletten Output
        
        Args:
            mermaid (str): Mermaid-Diagramm
            kg (Dict[str, Any]): Knowledge Graph (optional)
            requirements (List[Dict[str, Any]]: Liste der Requirements (optional)
        
        Returns:
            Dict[str, Any]: Validierungsergebnis
        """
        validation_config = get_validation_config()
        
        # Alle Validierungs-Checks durchführen
        results = {}
        
        if validation_config.get("syntax_check", True):
            results["syntax"] = await self.validate_syntax(mermaid)
        
        if validation_config.get("consistency_check", True) and kg:
            results["consistency"] = await self.validate_consistency(kg, mermaid)
        
        if validation_config.get("completeness_check", True) and kg:
            results["completeness"] = await self.validate_completeness(kg, mermaid)
        
        if validation_config.get("semantic_check", True) and kg:
            results["semantic"] = await self.validate_semantic(kg)
        
        if validation_config.get("cross_reference_check", True) and requirements:
            results["cross_reference"] = await self.validate_cross_reference(requirements)
        
        # Gesamtergebnis berechnen
        all_valid = all(result.get("valid", False) for result in results.values())
        all_issues = []
        for result in results.values():
            all_issues.extend(result.get("issues", []))
        
        return {
            "syntax_valid": results.get("syntax", {}).get("valid", False),
            "consistency_valid": results.get("consistency", {}).get("valid", False),
            "completeness_valid": results.get("completeness", {}).get("valid", False),
            "semantic_valid": results.get("semantic", {}).get("valid", False),
            "cross_reference_valid": results.get("cross_reference", {}).get("valid", False),
            "valid": all_valid,
            "issues": all_issues
        }
