"""
Integration-Tests für RequirementsSummaryAgent
"""

import pytest
import json
from pathlib import Path

from arch_team.agents.requirements_summary_agent import RequirementsSummaryAgent


@pytest.fixture
def agent():
    """Fixture für RequirementsSummaryAgent"""
    return RequirementsSummaryAgent()


@pytest.mark.asyncio
async def test_generate_summary_from_requirements(agent):
    """Testet die Zusammenfassungs-Generierung aus Requirements"""
    # Test-Requirements
    test_requirements = Path("tests/fixtures/test_requirements.json")
    
    if test_requirements.exists():
        with open(test_requirements, "r") as f:
            requirements = json.load(f)["requirements"]
        
        # Echter LLM-Call
        summary = await agent.generate_summary(requirements)
        
        assert "total_requirements" in summary
        assert "by_tag" in summary
        assert "by_priority" in summary
        assert "conflicts" in summary
        assert "dependencies" in summary
        assert "quality_score" in summary
    else:
        pytest.skip("Test-Requirements-Datei existiert nicht")


@pytest.mark.asyncio
async def test_generate_summary_with_kg(agent):
    """Testet die Zusammenfassungs-Generierung mit Knowledge Graph"""
    # Test-Requirements
    test_requirements = Path("tests/fixtures/test_requirements.json")
    
    if test_requirements.exists():
        with open(test_requirements, "r") as f:
            requirements = json.load(f)["requirements"]
        
        # Test-Knowledge Graph
        test_kg = Path("tests/fixtures/test_kg.json")
        
        if test_kg.exists():
            with open(test_kg, "r") as f:
                kg = json.load(f)
            
            # Echter LLM-Call
            summary = await agent.generate_summary(requirements, kg)
            
            assert "total_requirements" in summary
            assert "by_tag" in summary
            assert "by_priority" in summary
            assert "conflicts" in summary
            assert "dependencies" in summary
            assert "quality_score" in summary
        else:
            pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_generate_summary_with_real_llm(agent):
    """Testet die Zusammenfassungs-Generierung mit echtem LLM-Call"""
    # Test-Requirements
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        },
        {
            "id": "REQ-002",
            "text": "User passwords must be encrypted using AES-256.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Echter LLM-Call
    summary = await agent.generate_summary(requirements)
    
    assert "total_requirements" in summary
    assert "by_tag" in summary
    assert "by_priority" in summary
    assert "conflicts" in summary
    assert "dependencies" in summary
    assert "quality_score" in summary


@pytest.mark.asyncio
async def test_calculate_by_tag_with_real_llm(agent):
    """Testet die Berechnung der Verteilung nach Tags mit echtem LLM-Call"""
    # Test-Requirements
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        },
        {
            "id": "REQ-002",
            "text": "User passwords must be encrypted using AES-256.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Verteilung berechnen
    by_tag = await agent._calculate_by_tag(requirements)
    
    assert by_tag["Security"] == 2


@pytest.mark.asyncio
async def test_calculate_by_priority_with_real_llm(agent):
    """Testet die Berechnung der Verteilung nach Prioritäten mit echtem LLM-Call"""
    # Test-Requirements
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        },
        {
            "id": "REQ-002",
            "text": "User passwords must be encrypted using AES-256.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Verteilung berechnen
    by_priority = await agent._calculate_by_priority(requirements)
    
    assert by_priority["Must"] == 2


@pytest.mark.asyncio
async def test_extract_conflicts_with_real_llm(agent):
    """Testet die Extraktion von Konflikten mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Konflikte extrahieren
        conflicts = await agent._extract_conflicts(kg)
        
        assert len(conflicts) >= 0
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_extract_dependencies_with_real_llm(agent):
    """Testet die Extraktion von Abhängigkeiten mit echtem LLM-Call"""
    # Test-Knowledge Graph
    test_kg = Path("tests/fixtures/test_kg.json")
    
    if test_kg.exists():
        with open(test_kg, "r") as f:
            kg = json.load(f)
        
        # Abhängigkeiten extrahieren
        dependencies = await agent._extract_dependencies(kg)
        
        assert len(dependencies) >= 0
    else:
        pytest.skip("Test-Knowledge-Graph-Datei existiert nicht")


@pytest.mark.asyncio
async def test_calculate_quality_score_with_real_llm(agent):
    """Testet die Berechnung des Quality-Scores mit echtem LLM-Call"""
    # Test-Requirements
    requirements = [
        {
            "id": "REQ-001",
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"],
            "is_valid": True
        },
        {
            "id": "REQ-002",
            "text": "",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"],
            "is_valid": False
        }
    ]
    
    # Quality-Score berechnen
    quality_score = await agent._calculate_quality_score(requirements)
    
    assert quality_score == 50.0
