import pytest
import pandas as pd
import os
from src.titanic_app import TitanicAnalysis


class TestTitanicAnalysis:
    """Тесты для класса анализа данных Титаника."""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных."""
        data = {
            'PassengerId': [1, 2, 3, 4, 5, 6],
            'Survived': [1, 0, 1, 0, 1, 0],
            'Pclass': [1, 3, 1, 3, 2, 1],
            'Sex': ['female', 'male', 'female', 'male', 'female', 'male'],
            'Age': [22, 35, 28, 45, 31, 50],
            'Fare': [50.0, 10.0, 80.0, 15.0, 30.0, 60.0]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data, tmp_path):
        """Создание временного CSV файла."""
        file_path = tmp_path / "test_titanic.csv"
        sample_data.to_csv(file_path, index=False)
        return str(file_path)
    
    @pytest.fixture
    def analyzer(self, temp_csv_file):
        """Создание экземпляра анализатора с тестовыми данными."""
        return TitanicAnalysis(temp_csv_file)
    
    def test_data_loading(self, analyzer, sample_data):
        """Тест загрузки данных."""
        pd.testing.assert_frame_equal(analyzer.df, sample_data)
    
    def test_file_not_found(self):
        """Тест обработки отсутствующего файла."""
        with pytest.raises(FileNotFoundError):
            TitanicAnalysis("nonexistent_file.csv")
    
    def test_filter_by_sex(self, analyzer):
        """Тест фильтрации по полу."""
        result = analyzer.filter_data(sex='female')
        assert len(result) == 3
        assert all(result['Sex'] == 'female')
    
    def test_filter_by_survived(self, analyzer):
        """Тест фильтрации по выживанию."""
        result = analyzer.filter_data(survived=1)
        assert len(result) == 3
        assert all(result['Survived'] == 1)
    
    def test_filter_by_pclass(self, analyzer):
        """Тест фильтрации по классу."""
        result = analyzer.filter_data(pclass=[1, 2])
        assert len(result) == 4
        assert set(result['Pclass'].unique()) == {1, 2}
    
    def test_filter_by_fare_range(self, analyzer):
        """Тест фильтрации по диапазону платы."""
        result = analyzer.filter_data(fare_range=(20.0, 40.0))
        assert len(result) == 1
        assert result.iloc[0]['Fare'] == 30.0
    
    def test_combined_filters(self, analyzer):
        """Тест комбинированных фильтров."""
        result = analyzer.filter_data(
            sex='female',
            survived=1,
            pclass=[1],
            fare_range=(40.0, 100.0)
        )
        assert len(result) == 2
    
    def test_get_statistics(self, analyzer):
        """Тест получения статистики."""
        filtered_df = analyzer.filter_data(sex='female')
        stats = analyzer.get_statistics(filtered_df)
        
        assert stats['total_count'] == 3
        assert stats['average_fare'] == pytest.approx(53.33, rel=1e-2)
        assert stats['average_age'] == pytest.approx(27.0, rel=1e-2)


def test_run_streamlit_app(monkeypatch):
    """Тест запуска Streamlit приложения."""
    # Мокаем Streamlit функции чтобы избежать их реального выполнения
    mock_functions = {}
    for func in ['image', 'title', 'sidebar', 'selectbox', 'multiselect', 
                 'slider', 'dataframe', 'write', 'header']:
        mock_functions[func] = lambda *args, **kwargs: None
    
    for func_name, mock_func in mock_functions.items():
        monkeypatch.setattr(f"streamlit.{func_name}", mock_func)
    
    # Импортируем и запускаем функцию
    from src.titanic_app import run_streamlit_app
    run_streamlit_app()
