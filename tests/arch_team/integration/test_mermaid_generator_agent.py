"""
Integration-Tests für MermaidGeneratorAgent
"""

import pytest
import json
from pathlib import Path

from arch_team.agents.mermaid_generator_agent import MermaidGeneratorAgent


@pytest.fixture
def agent():
    """Fixture für MermaidGeneratorAgent"""
    return MermaidGeneratorAgent()


@pytest.mark.asyncio
async def test_generate_mermaid_diagram_from_kg(agent):
    """Testet die Mermaid-Diagramm-Generierung aus Knowledge Graph"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Echter LLM-Call
        mermaid_result = await agent.generate_mermaid_diagram(kg, "dependency")
        
        assert "mermaid" in mermaid_result
        assert "diagram_type" in mermaid_result
        assert "metadata" in mermaid_result
        assert "graph" in mermaid_result["mermaid"]
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_generate_dependency_diagram(agent):
    """Testet die Generierung eines Dependency-Diagramms"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Echter LLM-Call
        mermaid_result = await agent.generate_dependency_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "dependency"
        assert "graph" in mermaid_result["mermaid"]
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_generate_conflict_diagram(agent):
    """Testet die Generierung eines Conflict-Diagramms"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Echter LLM-Call
        mermaid_result = await agent.generate_conflict_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "conflict"
        assert "graph" in mermaid_result["mermaid"]
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_generate_grouped_diagram(agent):
    """Testet die Generierung eines Grouped-Diagramms"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Echter LLM-Call
        mermaid_result = await agent.generate_grouped_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "grouped"
        assert "subgraph" in mermaid_result["mermaid"]
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_generate_actor_diagram(agent):
    """Testet die Generierung eines Actor-Diagramms"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Echter LLM-Call
        mermaid_result = await agent.generate_actor_diagram(kg)
        
        assert mermaid_result["diagram_type"] == "actor"
        assert "graph" in mermaid_result["mermaid"]
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_validate_mermaid_syntax_with_real_llm(agent):
    """Testet die Mermaid-Syntax-Validierung mit echtem LLM-Call"""
    # Gültiges Mermaid-Diagramm
    valid_mermaid = "graph TD\n  REQ001[REQ-001: Auth] --> REQ002[REQ-002: Encryption]"
    
    # Syntax-Validierung
    result = await agent.validate_mermaid_syntax(valid_mermaid)
    
    assert result["valid"] == True
    assert len(result["issues"]) == 0


@pytest.mark.asyncio
async def test_generate_mermaid_diagram_with_real_llm(agent):
    """Testet die Mermaid-Diagramm-Generierung mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Echter LLM-Call
        mermaid_result = await agent.generate_mermaid_diagram(kg, "dependency")
        
        assert "mermaid" in mermaid_result
        assert "diagram_type" in mermaid_result
        assert "metadata" in mermaid_result
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")
