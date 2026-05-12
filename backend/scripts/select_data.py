import sqlite3
conn = sqlite3.connect('alloy_database.db')
cursor = conn.cursor()

# 检查有多少条记录有有效的抗拉强度数据
cursor.execute('SELECT COUNT(*) FROM mechanical_properties WHERE ultimate_tensile_strength IS NOT NULL AND ultimate_tensile_strength > 0')
uts_count = cursor.fetchone()[0]

# 检查有多少条记录有有效的延伸率数据
cursor.execute('SELECT COUNT(*) FROM mechanical_properties WHERE elongation IS NOT NULL AND elongation > 0')
elong_count = cursor.fetchone()[0]

# 检查同时有两个指标的记录
cursor.execute('''
    SELECT COUNT(*) 
    FROM mechanical_properties 
    WHERE ultimate_tensile_strength IS NOT NULL AND ultimate_tensile_strength > 0 
      AND elongation IS NOT NULL AND elongation > 0
''')
both_count = cursor.fetchone()[0]

print('=== 机械性能数据统计 ===')
print(f'总记录数: 107')
print(f'有抗拉强度数据的记录: {uts_count}')
print(f'有延伸率数据的记录: {elong_count}')
print(f'同时有两个指标的记录: {both_count}')

conn.close()