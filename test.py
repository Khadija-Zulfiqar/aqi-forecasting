# import hopsworks, os
# from dotenv import load_dotenv
# load_dotenv()
# project = hopsworks.login(
#     host=os.getenv('HOPSWORKS_HOST'),
#     api_key_value=os.getenv('HOPSWORKS_API_KEY'),
#     project=os.getenv('HOPSWORKS_PROJECT')
# )
# fs = project.get_feature_store()
# fg = fs.get_feature_group(name='aqi_features', version=1)
# df = fg.read()
# print(f'Rows available: {len(df)}')
# print(f'AQI range: {df["aqi"].min()} to {df["aqi"].max()}')
# print(df[['timestamp','aqi']].tail(5))


import requests, os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENWEATHER_API_KEY')
r = requests.get(f'http://api.openweathermap.org/data/2.5/air_pollution?lat=24.8607&lon=67.0011&appid={key}').json()
print('Raw OpenWeather response:')
print(r)
