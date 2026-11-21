import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import numpy as np
from phik import phik_matrix

class DataProcessor:
    def __init__(self):
        self.database_path = 'postgresql://{}:{}@{}:{}/{}'.format(
            os.getenv('DB_USER'),
            os.getenv('DB_PWD'),
            os.getenv('DB_HOST'),
            os.getenv('DB_PORT'),
            os.getenv('DB_NAME')
        )
        self.rates_path = "data/currency_rates.csv"
        self.data = None
        self.profile = None

    def load_data(self):
        query = '''
                -- Настройка параметра synchronize_seqscans важна для проверки
                WITH set_config_precode AS (SELECT set_config('synchronize_seqscans', 'off', true))
                -- Напишите ваш запрос ниже
                SELECT p.user_id, \
                       p.device_type_canonical, \
                       p.order_id, \
                       p.created_dt_msk                                                                                               as order_dt, \
                       p.created_ts_msk                                                                                               as order_ts, \
                       p.currency_code, \
                       p.revenue, \
                       p.tickets_count, \
                       (p.created_dt_msk::DATE - LAG(p.created_dt_msk::DATE) OVER (PARTITION BY p.user_id ORDER BY p.created_dt_msk)) as days_since_prev, \
                       p.event_id, \
                       e.event_name_code                                                                                              as event_name, \
                       p.service_name, \
                       e.event_type_main, \
                       r.region_name, \
                       c.city_name

                FROM afisha.purchases AS p
                         LEFT JOIN afisha.events AS e ON p.event_id = e.event_id
                         LEFT JOIN afisha.city AS c ON e.city_id = c.city_id
                         LEFT JOIN afisha.regions AS r ON c.region_id = r.region_id
                WHERE p.device_type_canonical IN ('mobile', 'desktop') \
                  AND e.event_type_main NOT LIKE 'фильм'
                ORDER BY p.user_id \
                '''
        engine = create_engine(self.database_path)
        self.data = pd.read_sql_query(query, engine)
        return self.data

    def load_rates_df(self):
        rates = pd.read_csv(self.rates_path)
        rates = rates.rename(columns={"data": "order_dt"})
        rates["order_dt"] = pd.to_datetime(rates["order_dt"], errors="coerce")
        return rates

    def preprocess(self):
        if self.data is None:
            self.load_data()

        df = self.data.copy()

        # Изменение столбцов
        df['days_since_prev'] = df['days_since_prev'].astype("float32")
        df['tickets_count'] = df['tickets_count'].astype("int16")

        # Работа с выручкой
        rates = self.load_rates_df()
        df = df.merge(rates[["order_dt", "curs"]], on="order_dt", how="left")

        df["revenue_rub"] = np.where(
            df["currency_code"].str.lower().eq("rub"),
            df["revenue"],
            (df["revenue"] / 100) * df["curs"]
        )

        # Удаление неявных дубликатов
        columns = [
            'user_id',
            'device_type_canonical',
            'order_ts',
            'currency_code',
            'revenue',
            'tickets_count',
            'event_id',
            'event_name',
            'service_name',
            'event_type_main',
            'region_name',
            'city_name'
        ]
        df = df.drop_duplicates(subset=columns, keep='first', inplace=False)

        df = df[df["revenue_rub"] >= 0]
        percent_99 = df["revenue_rub"].quantile(0.99)
        df = df[df["revenue_rub"] < percent_99]

        self.data = df
        return df

    def create_user_profiles(self):
        if self.data is None:
            self.preprocess()

        df = self.data.sort_values(by=['order_ts', 'region_name'])

        profile = df.groupby(by='user_id').agg(
            first_order_dt=('order_dt', 'min'),
            last_order_dt=('order_dt', 'max'),
            first_device=('device_type_canonical', 'first'),
            first_region=('region_name', 'first'),
            first_event_type=('event_type_main', 'first'),
            all_orders=('order_id', 'nunique'),
            sum_revenue=('revenue_rub', 'sum'),
        )

        profile['avg_revenue'] = profile['sum_revenue'] / profile['all_orders']

        profile['it_two'] = profile['all_orders'] >= 2

        profile['days_between_min_max'] = (profile['last_order_dt'] - profile['first_order_dt']).dt.days

        profile['avg_days_between_orders'] = np.where(profile['all_orders'] > 1,
                                                      profile['days_between_min_max'] / (profile['all_orders'] - 1),
                                                      np.nan)

        percent_99 = profile['avg_revenue'].quantile(0.99)
        profile = profile[profile['avg_revenue'] < percent_99]

        profile["first_order_weekdays"] = profile["first_order_dt"].dt.weekday
        self.profile = profile.reset_index()
        return self.profile

    def retention_by_segment(self, segment_column):
        if self.profile is None:
            self.create_user_profiles()

        result = (
            self.profile
            .groupby(segment_column, as_index=False)
            .agg(
                users=('user_id', 'nunique'),
                it_two=('it_two', 'mean'))
            .sort_values(by='users', ascending=False)
        )

        result['it_two'].plot(
            kind='bar',
            xlabel=segment_column,
            ylabel='Доля пользователей, совершивших 2 и более заказа',
            title=f'Доля повторных покупок по сегменту: {segment_column}',
            rot=45,
            legend=False,
            grid=True,
            figsize=(10, 6)
        )

        plt.tight_layout()
        plt.show()

        return result

    # Функция подтверждения
    def compare_hypotheses(self):
        if self.profile is None:
            self.create_user_profiles()

        sports_concerts = (
            self.profile
            .groupby('first_event_type', as_index=False)
            .agg(
                users=('user_id', 'nunique'),
                it_two=('it_two', 'mean'))
            .sort_values(by='it_two', ascending=False)
        )

        regions = (
            self.profile
            .groupby('first_region', dropna=False)
            .agg(
                users=('user_id', 'nunique'),
                it_two=('it_two', 'mean')
            )
            .sort_values('users', ascending=False)
        )

        sports_concerts['it_two'].plot(
            kind='bar',
            xlabel='Категория мероприятия',
            ylabel='Доля повторных покупок',
            title='Сравнение спорта и концертов',
            rot=0,
            legend=False,
            grid=True,
            figsize=(10, 6)
        )

        plt.tight_layout()
        plt.show()

        # График топ-10 регионов по числу пользователей
        top_regions = regions.head(10).sort_values('it_two')

        top_regions['it_two'].plot(
            kind='barh',
            ylabel='Регионы',
            xlabel='Доля пользователей, совершивших 2 и более заказа',
            title='Повторные покупки в топ-10 регионах по числу пользователей',
            legend=False,
            grid=True,
            figsize=(10, 6)
        )

        plt.tight_layout()
        plt.show()

        return {
            'sports_vs_concerts': sports_concerts,
            'regions_by_user_count': regions
        }

    def correlation_analysis(self):
        if self.profile is None:
            self.create_user_profiles()

        interval_cols = [
            'all_orders',
            'sum_revenue',
            'avg_revenue',
            'days_between_min_max',
            'avg_days_between_orders'
        ]

        data = self.profile

        corr_matrix = data.drop(columns=['user_id']).phik_matrix(interval_cols=interval_cols)

        total_orders_corr = (
            corr_matrix
            .loc[corr_matrix.index != 'all_orders'][['all_orders']]
            .sort_values('all_orders', ascending=False)
        )


        total_orders_corr.plot(
            kind='barh',
            figsize=(10, 6),
            xlabel='Коэффициент корреляции',
            ylabel='Признаки',
            title='Связь признаков с количеством заказов пользователя',
            grid=True,
            legend=False
        )

        plt.tight_layout()
        plt.show()

        return total_orders_corr