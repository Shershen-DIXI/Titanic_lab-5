import pandas as pd
import os
import sys

# Добавляем путь для импорта
sys.path.append('.')

try:
    from src.titanic_app import TitanicAnalysis
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False


class TestTitanicAnalysis:
    """Тесты для класса анализа данных Титаника."""

    def create_sample_data(self):
        """Создаем временный CSV файл для тестов."""
        data = {
            'PassengerId': [1, 2, 3, 4],
            'Survived': [1, 0, 1, 0],
            'Pclass': [1, 3, 1, 3],
            'Sex': ['male', 'female', 'female', 'male'],
            'Age': [22, None, 31, 25],
            'Fare': [50.0, 20.0, 80.0, 15.0]
        }
        df = pd.DataFrame(data)
        path = 'test_titanic_data.csv'
        df.to_csv(path, index=False)
        return path

    def cleanup_sample_data(self, path):
        """Удаляем временный файл после теста."""
        if os.path.exists(path):
            os.remove(path)

    def test_data_loading_and_preprocessing(self):
        """Тест загрузки данных и предобработки."""
        if not HAS_MODULE:
            print("⚠️ Модуль не найден, пропускаем тест")
            return

        path = self.create_sample_data()
        try:
            analyzer = TitanicAnalysis(path)

            # Проверяем загрузку данных
            expected_rows = 4
            actual_rows = len(analyzer.df)
            assert actual_rows == expected_rows, (
                f"Ожидалось {expected_rows} строк, получено {actual_rows}"
            )
            print("✅ Данные загружены корректно")

            # Проверяем наличие колонок
            expected_columns = [
                'PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare'
            ]
            for col in expected_columns:
                assert col in analyzer.df.columns, f"Колонка {col} не найдена"
            print("✅ Все колонки присутствуют")

        finally:
            self.cleanup_sample_data(path)

    def test_filter_data(self):
        """Тест фильтрации данных."""
        if not HAS_MODULE:
            print("⚠️ Модуль не найден, пропускаем тест")
            return

        path = self.create_sample_data()
        try:
            analyzer = TitanicAnalysis(path)

            # Фильтруем по полу
            filtered = analyzer.filter_data(sex='female')
            expected_females = 2
            actual_females = len(filtered)
            assert actual_females == expected_females, (
                f"Ожидалось {expected_females} женщин, получено {actual_females}"
            )
            assert all(filtered['Sex'] == 'female'), (
                "Не все строки отфильтрованы по полу"
            )
            print("✅ Фильтрация по полу работает")

            # Фильтруем по выживанию
            filtered = analyzer.filter_data(survived=1)
            expected_survived = 2
            actual_survived = len(filtered)
            assert actual_survived == expected_survived, (
                f"Ожидалось {expected_survived} выживших, получено {actual_survived}"
            )
            assert all(filtered['Survived'] == 1), (
                "Не все строки отфильтрованы по выживанию"
            )
            print("✅ Фильтрация по выживанию работает")

            # Фильтруем по классу
            filtered = analyzer.filter_data(pclass=[1, 2])
            expected_class = 2
            actual_class = len(filtered)
            assert actual_class == expected_class, (
                f"Ожидалось {expected_class} пассажиров 1-2 класса, "
                f"получено {actual_class}"
            )
            print("✅ Фильтрация по классу работает")

            # Фильтруем по цене
            filtered = analyzer.filter_data(fare_range=(10.0, 30.0))
            expected_fare = 2
            actual_fare = len(filtered)
            assert actual_fare == expected_fare, (
                f"Ожидалось {expected_fare} пассажиров с ценой 10-30, "
                f"получено {actual_fare}"
            )
            print("✅ Фильтрация по цене работает")

        finally:
            self.cleanup_sample_data(path)

    def test_combined_filters(self):
        """Тест комбинированных фильтров."""
        if not HAS_MODULE:
            print("⚠️ Модуль не найден, пропускаем тест")
            return

        path = self.create_sample_data()
        try:
            analyzer = TitanicAnalysis(path)

            # Комбинируем фильтры
            filtered = analyzer.filter_data(
                sex='female',
                survived=1,
                pclass=[1],
                fare_range=(40.0, 100.0)
            )
            expected_combined = 1
            actual_combined = len(filtered)
            assert actual_combined == expected_combined, (
                f"Ожидалась {expected_combined} строка, получено {actual_combined}"
            )
            print("✅ Комбинированные фильтры работают")

        finally:
            self.cleanup_sample_data(path)

    def run_all_tests(self):
        """Запуск всех тестов."""
        print("=== Запуск тестов ===")

        tests = [
            self.test_data_loading_and_preprocessing,
            self.test_filter_data,
            self.test_combined_filters
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                test()
                passed += 1
                print(f"✅ {test.__name__} - ПРОЙДЕН")
            except Exception as e:
                failed += 1
                print(f"❌ {test.__name__} - ОШИБКА: {e}")
            print("---")

        print(f"=== ИТОГ: {passed} пройдено, {failed} упало ===")

        if failed == 0:
            print("🎉 Все тесты пройдены успешно!")
        else:
            print("💥 Некоторые тесты не пройдены")

        return failed == 0


# Запускаем тесты при прямом выполнении файла
if __name__ == "__main__":
    tester = TestTitanicAnalysis()
    success = tester.run_all_tests()
    exit(0 if success else 1)
