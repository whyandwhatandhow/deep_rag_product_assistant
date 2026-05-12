import sqlite3
conn = sqlite3.connect('alloy_database.db')
cursor = conn.cursor()

tables = ['alloys', 'compositions', 'processing', 'microstructure', 'mechanical_properties', 'corrosion_properties']
print('=== 数据库记录统计 ===')
for table in tables:
    cursor.execute(f'SELECT COUNT(*) FROM {table}')
    count = cursor.fetchone()[0]
    print(f'{table}: {count} 条记录')

conn.close()