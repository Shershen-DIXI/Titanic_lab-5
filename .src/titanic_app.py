import pandas as pd

class TitanicAnalysis:
    """Класс для анализа данных о пассажирах Титаника."""
    
    def __init__(self, data_path='titanic_train.csv'):
        """Инициализация загрузкой данных."""
        self.df = self._load_data(data_path)
    
    def _load_data(self, data_path):
        """Загрузка данных из CSV файла."""
        try:
            return pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл '{data_path}' не найден")
    
    def filter_data(self, sex=None, survived=None, pclass=None, fare_range=None):
        """
        Фильтрация данных по заданным параметрам.
        
        Args:
            sex (str): Пол ('male' или 'female')
            survived (int): Выжил (1) или нет (0)
            pclass (list): Список классов
            fare_range (tuple): Диапазон платы за проезд (min, max)
        
        Returns:
            pd.DataFrame: Отфильтрованный DataFrame
        """
        filtered_df = self.df.copy()
        
        if sex:
            filtered_df = filtered_df[filtered_df['Sex'] == sex]
        if survived is not None:
            filtered_df = filtered_df[filtered_df['Survived'] == survived]
        if pclass:
            filtered_df = filtered_df[filtered_df['Pclass'].isin(pclass)]
        if fare_range:
            filtered_df = filtered_df[
                (filtered_df['Fare'] >= fare_range[0]) & 
                (filtered_df['Fare'] <= fare_range[1])
            ]
        
        return filtered_df
    
    def get_statistics(self, filtered_df):
        """Получение статистики по отфильтрованным данным."""
        return {
            'total_count': len(filtered_df),
            'average_fare': filtered_df['Fare'].mean(),
            'average_age': filtered_df['Age'].mean()
        }


def run_streamlit_app():
    """Запуск Streamlit приложения."""
    import streamlit as st
    
    st.image("titanic.jpg")
    st.title("Пассажиры Титаника")
    
    # Инициализация анализатора
    analyzer = TitanicAnalysis()
    
    st.sidebar.header("Фильтры")
    
    # Фильтр пола
    sex_filter = st.sidebar.selectbox(
        "Пол:",
        options=['female', 'male'],
        index=0
    )
    
    # Фильтр выживания
    survived_filter = st.sidebar.selectbox(
        "Выжил:",
        options=[1, 0],
        format_func=lambda x: "Да" if x == 1 else "Нет",
        index=0
    )
    
    # Фильтр класса
    class_filter = st.sidebar.multiselect(
        "Класс:",
        options=sorted(analyzer.df['Pclass'].unique()),
        default=sorted(analyzer.df['Pclass'].unique())
    )
    
    # Ползунок платы за проезд
    fare_range = st.sidebar.slider(
        "Плата за проезд:",
        min_value=float(analyzer.df['Fare'].min()),
        max_value=float(analyzer.df['Fare'].max()),
        value=(float(analyzer.df['Fare'].min()), float(analyzer.df['Fare'].max()))
    )
    
    # Применение фильтров
    filtered_df = analyzer.filter_data(
        sex=sex_filter,
        survived=survived_filter,
        pclass=class_filter,
        fare_range=fare_range
    )
    
    # Отображение результатов
    st.dataframe(filtered_df, use_container_width=True)
    st.write(f"Найдено записей: {len(filtered_df)}")
    
    # Дополнительная статистика
    stats = analyzer.get_statistics(filtered_df)
    st.sidebar.header("Статистика")
    st.sidebar.write(f"Всего записей: {stats['total_count']}")
    if stats['average_fare'] is not None:
        st.sidebar.write(f"Средняя плата: {stats['average_fare']:.2f}")
    if stats['average_age'] is not None:
        st.sidebar.write(f"Средний возраст: {stats['average_age']:.1f}")

if __name__ == "__main__":
    run_streamlit_app()
