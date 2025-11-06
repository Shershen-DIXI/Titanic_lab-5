import pandas as pd
import os
from titanic_analysis import TitanicAnalysis

class TestTitanicAnalysis:
    """Тесты для класса анализа данных Титаника."""
    
    @pytest.fixture
    def sample_data_path(self):
        # Создаем временный CSV файл для тестов
        data = {
            'PassengerId': [1, 2, 3, 4],
            'Survived': [1, 0, 1, 0],
            'Pclass': [1, 3, 1, 3],
            'Sex': ['male', 'female', 'female', 'male'],
            'Age': [22, None, 31, 25]  # Один пропущенный возраст
        }
        df = pd.DataFrame(data)
        path = 'test_titanic_data.csv'
        df.to_csv(path, index=False)
        yield path
        # Удаляем файл после теста
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def analysis_instance(self, sample_data_path):
        """Фикстура, создающая экземпляр класса для тестов."""
        return TitanicAnalysis(sample_data_path)
    
    def test_data_loading_and_preprocessing(self, analysis_instance):
        """Тест загрузки данных и предобработки (заполнение пропусков в возрасте)."""
        # Проверяем, что пропущенный возраст был заполнен медианой
        # Медиана возрастов [22, 31, 25] = 26.0
        assert analysis_instance.df['Age'].isnull().sum() == 0
        # Проверяем, что значение для пассажира 2 было заполнено правильно
        # В реальном тесте нужно аккуратно проверить конкретную строку
        assert analysis_instance.df.loc[1, 'Age'] == 26.0
    
    def test_get_survival_statistics_by_sex(self, analysis_instance):
        """Тест расчета статистики выживаемости по полу."""
        result = analysis_instance.get_survival_statistics('Sex')
        
        # Ожидаемые результаты на основе наших тестовых данных
        expected_data = {
            'Sex': ['female', 'male'],
            'total': [2, 2],
            'survived': [1, 1],  # Одна женщина выжила, один мужчина выжил
            'survival_rate': [0.5, 0.5]
        }
        expected_df = pd.DataFrame(expected_data)
        
        pd.testing.assert_frame_equal(result, expected_df)
    
    def test_get_survival_statistics_invalid_column(self, analysis_instance):
        """Тест на вызов ошибки при неверном имени столбца."""
        with pytest.raises(ValueError, match="Столбец 'InvalidColumn' не найден в данных."):
            analysis_instance.get_survival_statistics('InvalidColumn')
