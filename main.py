from dotenv import load_dotenv
from src.processing import DataProcessor

def main():
    load_dotenv()

    processor = DataProcessor()

    print('Загрузка и предобработка данных')

    tmp = processor.load_data()
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
    print(hypothesis_result['regions_by_user_count'].head(10).to_string())

    correlations = processor.correlation_analysis()

    print('\nСвязь признаков с количеством заказов:')
    print(correlations.to_string())

    processor.print_report(
        tmp=tmp,
        profiles=profile,
        event_retention=event_retention,
        device_retention=device_retention,
        region_retention=region_retention,
        correlations=correlations
    )

if __name__ == '__main__':
    main()