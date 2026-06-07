import hopsworks, os
from dotenv import load_dotenv
load_dotenv()
project = hopsworks.login(
    host=os.getenv('HOPSWORKS_HOST'),
    api_key_value=os.getenv('HOPSWORKS_API_KEY'),
    project=os.getenv('HOPSWORKS_PROJECT')
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name='aqi_features', version=1)
df = fg.read()
print(f'Rows available: {len(df)}')
print(f'AQI range: {df["aqi"].min()} to {df["aqi"].max()}')
print(df[['timestamp','aqi']].tail(5))