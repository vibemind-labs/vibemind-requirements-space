"""
RequirementsExtractionAgent mit AutoGen 4.7.2
Extrahiert Requirements aus Dokumenten mit Cross-Chunk Kontext und Deduplizierung
"""

import os
import json
from typing import Dict, Any, List
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config.autogen_config import get_autogen_config, get_agent_config


class RequirementsExtractionAgent:
    """
    RequirementsExtractionAgent mit AutoGen 4.7.2
    Extrahiert Requirements aus Dokumenten mit Cross-Chunk Kontext und Deduplizierung
    """
    
    def __init__(self):
        """
        Initialisiert den RequirementsExtractionAgent
        """
        self.config = get_agent_config("requirements_extraction")
        
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
            name="requirements_extraction_assistant",
            system_message="""You are a requirements extraction expert. Your task is to extract requirements from the given text.
            
            For each requirement, extract:
            - id: A unique identifier (e.g., REQ-001, REQ-002)
            - text: The exact requirement text
            - tag: The category (e.g., Security, Functional, Performance)
            - priority: The priority level (Must, Should, May)
            - evidence_refs: References to the source chunks
            
            Return the requirements in JSON format.""",
            model_client=self.model_client,
            reflect_on_tool_use=True
        )
    
    async def extract_requirements(self, text: str, chunk_id: str = "chunk-1") -> List[Dict[str, Any]]:
        """
        Extrahiert Requirements aus dem gegebenen Text
        
        Args:
            text (str): Der Text aus dem Requirements extrahiert werden sollen
            chunk_id (str): Die ID des Chunks
        
        Returns:
            List[Dict[str, Any]]: Liste der extrahierten Requirements
        """
        prompt = f"""Extract requirements from the following text:
        
        {text}
        
        Use chunk ID: {chunk_id} for evidence_refs."""
        
        cancellation_token = CancellationToken()
        response = await self.assistant.on_messages(
            [TextMessage(content=prompt, source="user")],
            cancellation_token
        )
        
        # Extrahierte Requirements aus der Antwort parsen
        requirements = self._parse_requirements(response.chat_message.content)
        
        return requirements
    
    async def extract_requirements_with_context(self, chunks: List[str], neighbor_refs: bool = True) -> List[Dict[str, Any]]:
        """
        Extrahiert Requirements aus Chunks mit Cross-Chunk Kontext
        
        Args:
            chunks (List[str]): Liste der Chunks
            neighbor_refs (bool): Ob Nachbar-Chunks als Kontext verwendet werden sollen
        
        Returns:
            List[Dict[str, Any]]: Liste der extrahierten Requirements
        """
        all_requirements = []
        
        for i, chunk in enumerate(chunks):
            # Cross-Chunk Kontext erstellen
            context = self._create_cross_chunk_context(chunks, i, neighbor_refs)
            
            # Requirements extrahieren
            requirements = await self.extract_requirements(context, f"chunk-{i+1}")
            
            # Requirements zur Liste hinzufügen
            all_requirements.extend(requirements)
        
        # Deduplizierung durchführen
        deduplicated_requirements = await self.deduplicate_requirements(all_requirements)
        
        return deduplicated_requirements
    
    def _create_cross_chunk_context(self, chunks: List[str], current_index: int, neighbor_refs: bool) -> str:
        """
        Erstellt Cross-Chunk Kontext für den aktuellen Chunk
        
        Args:
            chunks (List[str]): Liste aller Chunks
            current_index (int): Index des aktuellen Chunks
            neighbor_refs (bool): Ob Nachbar-Chunks als Kontext verwendet werden sollen
        
        Returns:
            str: Cross-Chunk Kontext
        """
        if not neighbor_refs:
            return chunks[current_index]
        
        # Vorherige und nachfolgende Chunks als Kontext hinzufügen
        context_parts = []
        
        # Vorheriger Chunk
        if current_index > 0:
            context_parts.append(f"[Previous Chunk]: {chunks[current_index - 1]}")
        
        # Aktueller Chunk
        context_parts.append(f"[Current Chunk]: {chunks[current_index]}")
        
        # Nachfolgender Chunk
        if current_index < len(chunks) - 1:
            context_parts.append(f"[Next Chunk]: {chunks[current_index + 1]}")
        
        return "\n\n".join(context_parts)
    
    async def deduplicate_requirements(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dedupliziert Requirements basierend auf semantischer Ähnlichkeit
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
        
        Returns:
            List[Dict[str, Any]]: Deduplizierte Liste der Requirements
        """
        # Einfache Deduplizierung basierend auf exaktem Text-Abgleich
        seen_texts = set()
        deduplicated = []
        
        for req in requirements:
            text = req["text"].strip().lower()
            
            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(req)
        
        return deduplicated
    
    async def validate_requirements(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validiert Requirements gegen Regeln
        
        Args:
            requirements (List[Dict[str, Any]]): Liste der Requirements
        
        Returns:
            List[Dict[str, Any]]: Validierte Liste der Requirements
        """
        validated = []
        
        for req in requirements:
            # Validierungsregeln
            is_valid = True
            issues = []
            
            # Prüfen, ob alle Pflichtfelder vorhanden sind
            required_fields = ["id", "text", "tag", "priority", "evidence_refs"]
            for field in required_fields:
                if field not in req:
                    is_valid = False
                    issues.append(f"Missing required field: {field}")
            
            # Prüfen, ob der Text nicht leer ist
            if not req.get("text", "").strip():
                is_valid = False
                issues.append("Requirement text is empty")
            
            # Prüfen, ob die Priorität gültig ist
            valid_priorities = ["Must", "Should", "May"]
            if req.get("priority") not in valid_priorities:
                is_valid = False
                issues.append(f"Invalid priority: {req.get('priority')}")
            
            # Validierungsergebnis hinzufügen
            req["is_valid"] = is_valid
            req["validation_issues"] = issues
            
            validated.append(req)
        
        return validated
    
    def _parse_requirements(self, response: str) -> List[Dict[str, Any]]:
        """
        Parst Requirements aus der LLM-Antwort
        
        Args:
            response (str): LLM-Antwort
        
        Returns:
            List[Dict[str, Any]]: Liste der Requirements
        """
        try:
            # Versuchen, JSON aus der Antwort zu extrahieren
            # Dies ist eine vereinfachte Implementierung
            # In der Praxis sollte dies robuster sein
            if "extract_requirements" in response:
                # Function-Calling-Antwort parsen
                import re
                match = re.search(r'"requirements":\s*\[(.*?)\]', response, re.DOTALL)
                if match:
                    requirements_json = f"[{match.group(1)}]"
                    return json.loads(requirements_json)
            
            # Fallback: Versuchen, JSON direkt zu parsen
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Leere Liste zurückgeben
            return []
    
    async def extract_requirements_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extrahiert Requirements aus einer Datei
        
        Args:
            file_path (str): Pfad zur Datei
        
        Returns:
            List[Dict[str, Any]]: Liste der extrahierten Requirements
        """
        # Datei lesen
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Requirements extrahieren
        requirements = await self.extract_requirements(text, "file-1")
        
        # Requirements validieren
        validated_requirements = await self.validate_requirements(requirements)
        
        return validated_requirements
