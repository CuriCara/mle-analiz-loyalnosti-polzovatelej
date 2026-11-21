from dotenv import load_dotenv
from src.processing import DataProcessor

def main():
    load_dotenv()

    processor = DataProcessor()

    print('Загрузка и предобработка данных завершены')

    processor.load_data()
    data = processor.preprocess()

    print(f'Количество строк после обработки: {len(data)}')
    print(f'Количество пользователей: {data["user_id"].nunique()}')

    print('\nСоздание профилей')

    profile = processor.create_user_profiles()

    print(f'Количество пользователей в профиле: {len(profile)}')
    print(f'Доля повторных покупателей: {profile["it_two"].mean():.2%}')

    event_retention = processor.retention_by_segment('first_event_type')
    print('\nПовторные покупки по типам мероприятий:')
    print(event_retention.to_string(index=False))

    device_retention = processor.retention_by_segment('first_device')
    print('\nПовторные покупки по типам мероприятий:')
    print(device_retention.to_string(index=False))

    region_retention = processor.retention_by_segment('first_region')
    print('\nПовторные покупки по типам мероприятий:')
    print(region_retention.to_string(index=False))\

    hypothesis_result = processor.compare_hypotheses()

    print('\nПроверка гипотезы по типам мероприятий:')
    print(hypothesis_result['sports_vs_concerts'].to_string(index=False))

    print('\nПроверка гипотезы по регионам:')
    print( hypothesis_result['regions_by_user_count'].head(10).to_string())

    correlations = processor.correlation_analysis()

    print('\nСвязь признаков с количеством заказов:')
    print(correlations.to_string())

    print("Общие выводы и рекомендации"
          "В ходе работы было проанализировано 290611 записей и 15 колонок о заказах на Яндекс Афише. При обработке данных было отфильтровано 3428 записей. Так же выручка с заказов была приведена к единой валюте RUB из KTZ по актуальному курсу на то время. Был составлен профиль пользователя, содержащий данные покупках пользователя и первой покупке."
          "В ходе анализа выявлено, что на повторную покупку пользователя влияет регион покупки, а так же категория. Больше всего люди повторяли покупки в `Шанырский регион`(0.68), а так же меньше всего повторяли покупки люди после покупки билета на категорию мероприятия `спорт` (0.56). Так же чаще всего возвращались к покупке пользователи, которые делали первую покупку в субботу (0.64). На количество же заказов больше всего влияет средняя выручка sum_revenue (0.99)."
          "Рекомендации по привлечению пользователей к повторной покупке:"
          "- Для регионов, где малая доля повторных покупок у пользователей, предлагать персональные скидки и предложения;"
          "- Для пользователей, купивших билеты на мероприятия категории `ёлки` и `спорт` создать персональные предложения на вторую покупку;"
          " Так как большая часть пользователей совершают покупку через `mobile` устройство (0.82), но доля повторных покупок меньше чем у `desctop`(0.61 и 0.64 соответственно), то стоит сделать скидки и предложения для мобильных приложения на повторные покупки.")

    if __name__ == '__main__':
        main()