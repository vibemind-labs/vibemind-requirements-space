"""
Test-Script zur Verifizierung der AutoGen 4.7.2 Installation
"""

import os
import sys
import json
from pathlib import Path


def test_autogen_version():
    """Testet die AutoGen-Version"""
    try:
        import autogen
        print(f"✓ AutoGen-Version: {autogen.__version__}")
        assert autogen.__version__ == "4.7.2", f"AutoGen-Version ist {autogen.__version__}, erwartet 4.7.2"
        return True
    except ImportError as e:
        print(f"✗ AutoGen konnte nicht importiert werden: {e}")
        return False
    except AssertionError as e:
        print(f"✗ {e}")
        return False


def test_autogen_config():
    """Testet die AutoGen-Konfiguration"""
    try:
        from config.autogen_config import get_autogen_config
        config = get_autogen_config()
        print(f"✓ AutoGen-Konfiguration geladen")
        assert config["model"] is not None, "Model ist nicht definiert"
        assert config["api_key"] is not None, "API-Key ist nicht definiert"
        print(f"✓ Model: {config['model']}")
        print(f"✓ API-Key: {config['api_key'][:10]}...")
        return True
    except ImportError as e:
        print(f"✗ AutoGen-Konfiguration konnte nicht importiert werden: {e}")
        return False
    except AssertionError as e:
        print(f"✗ {e}")
        return False


def test_autogen_config_file():
    """Testet die AutoGen-Konfigurationsdatei"""
    try:
        config_path = Path("config/autogen_config.json")
        assert config_path.exists(), f"Konfigurationsdatei {config_path} existiert nicht"
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"✓ AutoGen-Konfigurationsdatei gefunden")
        assert "model" in config, "Model ist nicht in der Konfiguration definiert"
        assert "api_key" in config, "API-Key ist nicht in der Konfiguration definiert"
        assert "agents" in config, "Agents sind nicht in der Konfiguration definiert"
        assert "chunking" in config, "Chunking ist nicht in der Konfiguration definiert"
        assert "diagram_types" in config, "Diagram-Typen sind nicht in der Konfiguration definiert"
        assert "relationship_types" in config, "Beziehungstypen sind nicht in der Konfiguration definiert"
        assert "node_types" in config, "Node-Typen sind nicht in der Konfiguration definiert"
        assert "validation" in config, "Validierung ist nicht in der Konfiguration definiert"
        
        print(f"✓ Alle Konfigurationsfelder sind vorhanden")
        return True
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"✗ Fehler beim Laden der Konfigurationsdatei: {e}")
        return False


def test_environment_variables():
    """Testet die Environment-Variablen"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")
        
        if openai_api_key:
            print(f"✓ OPENAI_API_KEY ist gesetzt: {openai_api_key[:10]}...")
        else:
            print(f"⚠ OPENAI_API_KEY ist nicht gesetzt")
        
        if openai_api_base:
            print(f"✓ OPENAI_API_BASE ist gesetzt: {openai_api_base}")
        else:
            print(f"⚠ OPENAI_API_BASE ist nicht gesetzt")
        
        return True
    except Exception as e:
        print(f"✗ Fehler beim Prüfen der Environment-Variablen: {e}")
        return False


def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("AutoGen 4.7.2 Installationstest")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: AutoGen-Version
    print("Test 1: AutoGen-Version")
    print("-" * 60)
    results.append(test_autogen_version())
    print()
    
    # Test 2: AutoGen-Konfigurationsdatei
    print("Test 2: AutoGen-Konfigurationsdatei")
    print("-" * 60)
    results.append(test_autogen_config_file())
    print()
    
    # Test 3: AutoGen-Konfiguration
    print("Test 3: AutoGen-Konfiguration")
    print("-" * 60)
    results.append(test_autogen_config())
    print()
    
    # Test 4: Environment-Variablen
    print("Test 4: Environment-Variablen")
    print("-" * 60)
    results.append(test_environment_variables())
    print()
    
    # Zusammenfassung
    print("=" * 60)
    print("Zusammenfassung")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Bestanden: {passed}/{total}")
    
    if passed == total:
        print("✓ Alle Tests bestanden!")
        return 0
    else:
        print(f"✗ {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(main())
