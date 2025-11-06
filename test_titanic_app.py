import pytest
import pandas as pd

def test_basic():
    """Базовый тест."""
    assert 1 + 1 == 2

def test_import():
    """Тест импорта основного модуля."""
    try:
        from src.titanic_app import TitanicAnalysis
        assert True
    except ImportError as e:
        assert False, f"Ошибка импорта: {e}"

class TestTitanicAnalysis:
    def test_creation(self, tmp_path):
        from src.titanic_app import TitanicAnalysis
        # Создаем тестовые данные
        data = {
            'PassengerId': [1, 2],
            'Survived': [1, 0],
            'Pclass': [1, 3],
            'Sex': ['female', 'male'],
            'Fare': [50.0, 10.0]
        }
        df = pd.DataFrame(data)
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        
        # Тестируем
        analyzer = TitanicAnalysis(str(file_path))
        assert len(analyzer.df) == 2
