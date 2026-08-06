"""
Integration-Tests für RequirementsExtractionAgent
"""

import pytest
import json
from pathlib import Path

from arch_team.agents.requirements_extraction_agent import RequirementsExtractionAgent


@pytest.fixture
def agent():
    """Fixture für RequirementsExtractionAgent"""
    return RequirementsExtractionAgent()


@pytest.mark.asyncio
async def test_extract_requirements_from_pdf(agent):
    """Testet die Requirements-Extraktion aus PDF"""
    # Test-PDF-Datei
    test_pdf = Path("data/test_documents/test_requirements.pdf")
    
    if test_pdf.exists():
        requirements = await agent.extract_requirements_from_file(str(test_pdf))
        
        assert len(requirements) > 0
        assert all("text" in req for req in requirements)
        assert all("tag" in req for req in requirements)
        assert all("priority" in req for req in requirements)
    else:
        pytest.skip("Test-PDF-Datei existiert nicht")


@pytest.mark.asyncio
async def test_extract_requirements_from_docx(agent):
    """Testet die Requirements-Extraktion aus DOCX"""
    # Test-DOCX-Datei
    test_docx = Path("data/test_documents/test_requirements.docx")
    
    if test_docx.exists():
        requirements = await agent.extract_requirements_from_file(str(test_docx))
        
        assert len(requirements) > 0
        assert all("text" in req for req in requirements)
        assert all("tag" in req for req in requirements)
        assert all("priority" in req for req in requirements)
    else:
        pytest.skip("Test-DOCX-Datei existiert nicht")


@pytest.mark.asyncio
async def test_extract_requirements_from_md(agent):
    """Testet die Requirements-Extraktion aus MD"""
    # Test-MD-Datei
    test_md = Path("data/test_documents/test_requirements.md")
    
    if test_md.exists():
        requirements = await agent.extract_requirements_from_file(str(test_md))
        
        assert len(requirements) > 0
        assert all("text" in req for req in requirements)
        assert all("tag" in req for req in requirements)
        assert all("priority" in req for req in requirements)
    else:
        pytest.skip("Test-MD-Datei existiert nicht")


@pytest.mark.asyncio
async def test_extract_requirements_from_txt(agent):
    """Testet die Requirements-Extraktion aus TXT"""
    # Test-TXT-Datei
    test_txt = Path("data/test_documents/test_requirements.txt")
    
    if test_txt.exists():
        requirements = await agent.extract_requirements_from_file(str(test_txt))
        
        assert len(requirements) > 0
        assert all("text" in req for req in requirements)
        assert all("tag" in req for req in requirements)
        assert all("priority" in req for req in requirements)
    else:
        pytest.skip("Test-TXT-Datei existiert nicht")


@pytest.mark.asyncio
async def test_extract_requirements_with_real_llm(agent):
    """Testet die Requirements-Extraktion mit echtem LLM-Call"""
    # Test-Text
    text = "The system must authenticate users within 2 seconds."
    
    # Echter LLM-Call
    requirements = await agent.extract_requirements(text)
    
    assert len(requirements) > 0
    assert requirements[0]["text"] == text


@pytest.mark.asyncio
async def test_extract_requirements_with_context_real_llm(agent):
    """Testet die Requirements-Extraktion mit Cross-Chunk Kontext und echtem LLM-Call"""
    # Test-Chunks
    chunks = [
        "The system must authenticate users within 2 seconds.",
        "User passwords must be encrypted using AES-256.",
        "The system shall support multi-factor authentication."
    ]
    
    # Echter LLM-Call mit Cross-Chunk Kontext
    requirements = await agent.extract_requirements_with_context(chunks, neighbor_refs=True)
    
    assert len(requirements) > 0
    assert all("text" in req for req in requirements)
    assert all("tag" in req for req in requirements)
    assert all("priority" in req for req in requirements)


@pytest.mark.asyncio
async def test_deduplication_real_llm(agent):
    """Testet die Deduplizierung mit echtem LLM-Call"""
    # Test-Requirements mit Duplikaten
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
            "text": "The system must authenticate users within 2 seconds.",
            "tag": "Security",
            "priority": "Must",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Deduplizierung
    deduplicated = await agent.deduplicate_requirements(requirements)
    
    assert len(deduplicated) == 1
    assert deduplicated[0]["id"] == "REQ-001"


@pytest.mark.asyncio
async def test_validation_real_llm(agent):
    """Testet die Validierung mit echtem LLM-Call"""
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
            "text": "",
            "tag": "Security",
            "priority": "Invalid",
            "evidence_refs": ["chunk-1"]
        }
    ]
    
    # Validierung
    validated = await agent.validate_requirements(requirements)
    
    assert len(validated) == 2
    assert validated[0]["is_valid"] == True
    assert validated[1]["is_valid"] == False
    assert len(validated[1]["validation_issues"]) > 0
