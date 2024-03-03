from influxdb import InfluxDBClient
import pandas as pd

# Connect to InfluxDB
client = InfluxDBClient(
    host='localhost', 
    port=32794, 
    username='', 
    password='', 
    database='nso')

query = """
    SELECT * FROM "span" 
    WHERE "name" = 'transaction' AND tid != 0 
    ORDER BY time DESC LIMIT 52"""

# Execute the query
result = client.query(query)
df = pd.DataFrame(result.get_points())

# Select only 'time', 'duration', and 'tid' columns
df = df[['time', 'duration', 'tid']]

# Print raw influx db data
# Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)

# Close the connection
client.close()