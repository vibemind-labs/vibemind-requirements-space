"""
Test-Skript für AutoGen 4.7.2 Importe
"""

import sys
import os

# Pfad zum Projekt hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_autogen_imports():
    """Testet die AutoGen 4.7.2 Importe"""
    print("Teste AutoGen 4.7.2 Importe...")
    
    try:
        from autogen_agentchat.agents import AssistantAgent
        print("[OK] AssistantAgent import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] AssistantAgent import fehlgeschlagen: {e}")
        return False
    
    try:
        from autogen_agentchat.messages import TextMessage
        print("[OK] TextMessage import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] TextMessage import fehlgeschlagen: {e}")
        return False
    
    try:
        from autogen_core import CancellationToken
        print("[OK] CancellationToken import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] CancellationToken import fehlgeschlagen: {e}")
        return False
    
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        print("[OK] OpenAIChatCompletionClient import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] OpenAIChatCompletionClient import fehlgeschlagen: {e}")
        return False
    
    print("\nAlle AutoGen 4.7.2 Importe erfolgreich!")
    return True


def test_agent_imports():
    """Testet die Agent-Importe"""
    print("\nTeste Agent-Importe...")
    
    try:
        from arch_team.agents.requirements_extraction_agent import RequirementsExtractionAgent
        print("[OK] RequirementsExtractionAgent import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] RequirementsExtractionAgent import fehlgeschlagen: {e}")
        return False
    
    try:
        from arch_team.agents.kg_relationship_agent import KGRelationshipAgent
        print("[OK] KGRelationshipAgent import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] KGRelationshipAgent import fehlgeschlagen: {e}")
        return False
    
    try:
        from arch_team.agents.mermaid_generator_agent import MermaidGeneratorAgent
        print("[OK] MermaidGeneratorAgent import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] MermaidGeneratorAgent import fehlgeschlagen: {e}")
        return False
    
    try:
        from arch_team.agents.requirements_summary_agent import RequirementsSummaryAgent
        print("[OK] RequirementsSummaryAgent import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] RequirementsSummaryAgent import fehlgeschlagen: {e}")
        return False
    
    try:
        from arch_team.agents.output_validator import OutputValidator
        print("[OK] OutputValidator import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] OutputValidator import fehlgeschlagen: {e}")
        return False
    
    print("\nAlle Agent-Importe erfolgreich!")
    return True


def test_config_imports():
    """Testet die Konfigurations-Importe"""
    print("\nTeste Konfigurations-Importe...")
    
    try:
        from config.autogen_config import get_autogen_config, get_agent_config
        print("[OK] autogen_config import erfolgreich")
    except ImportError as e:
        print(f"[FAIL] autogen_config import fehlgeschlagen: {e}")
        return False
    
    print("\nKonfigurations-Importe erfolgreich!")
    return True


if __name__ == "__main__":
    success = True
    
    success = test_autogen_imports() and success
    success = test_agent_imports() and success
    success = test_config_imports() and success
    
    if success:
        print("\n" + "="*50)
        print("Alle Tests erfolgreich! Das System funktioniert.")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("Einige Tests sind fehlgeschlagen.")
        print("="*50)
        sys.exit(1)
