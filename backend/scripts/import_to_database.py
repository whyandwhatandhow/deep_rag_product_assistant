import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
from app.database.alloy_db import AlloyDatabase
from typing import List, Dict, Any

def load_json_data(json_file: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {json_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []

def validate_and_fix_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证和修复数据"""
    fixed_data = data.copy()

    # 处理 None 值，确保 composition 和 mechanical_properties 不为 None
    if fixed_data.get('composition') is None:
        fixed_data['composition'] = {}
    if fixed_data.get('mechanical_properties') is None:
        fixed_data['mechanical_properties'] = {}
    if fixed_data.get('processing') is None:
        fixed_data['processing'] = {}
    if fixed_data.get('microstructure') is None:
        fixed_data['microstructure'] = {}

    if not fixed_data.get('alloy_id'):
        components = []
        comp = fixed_data.get('composition', {})
        if comp and comp.get('zinc'):
            components.append(f"Zn{comp['zinc']:.1f}")
        if comp and comp.get('magnesium'):
            components.append(f"Mg{comp['magnesium']:.1f}")
        if comp and comp.get('aluminum'):
            components.append(f"Al{comp['aluminum']:.1f}")

        year = fixed_data.get('year', 'unknown')
        base_id = ''.join(components) if components else 'Alloy'
        fixed_data['alloy_id'] = f"{base_id}_{year}_001"

    if fixed_data.get('composition') and isinstance(fixed_data['composition'], dict):
        comp = fixed_data['composition']
        for key in ['zinc', 'magnesium', 'aluminum', 'copper', 'silver']:
            if comp.get(key) and isinstance(comp[key], str):
                try:
                    comp[key] = float(comp[key])
                except ValueError:
                    comp[key] = None

    if fixed_data.get('mechanical_properties') and isinstance(fixed_data['mechanical_properties'], dict):
        mech = fixed_data['mechanical_properties']
        for key in ['ultimate_tensile_strength', 'yield_strength', 'elongation', 'hardness',
                    'elastic_modulus', 'fatigue_strength', 'impact_energy']:
            if mech.get(key) and isinstance(mech[key], str):
                try:
                    mech[key] = float(mech[key])
                except ValueError:
                    mech[key] = None

    return fixed_data

def main():
    print("\n=== 合金数据导入工具 ===\n")

    # 加载JSON数据 - 使用最新的 batch_alloy_data.json
    # 首先检查上级目录（项目根目录）
    json_file = "../batch_alloy_data.json"
    if not os.path.exists(json_file):
        json_file = "batch_alloy_data.json"
    if not os.path.exists(json_file):
        json_file = "optimized_alloy_data.json"

    if not os.path.exists(json_file):
        print(f"错误: 找不到数据文件")
        print(f"请确保 batch_alloy_data.json 或 optimized_alloy_data.json 存在")
        return

    print(f"从 {json_file} 加载数据...")

    data_list = load_json_data(json_file)
    if not data_list:
        print("没有数据可导入")
        return

    print(f"加载了 {len(data_list)} 条数据")

    db = AlloyDatabase("alloy_database.db")
    print(f"数据库已创建: alloy_database.db")

    validated_data = []
    for data in data_list:
        validated_data.append(validate_and_fix_data(data))

    print(f"\n开始导入数据...")
    success_count, fail_count = db.batch_import(validated_data)

    print(f"\n导入完成!")
    print(f"  成功: {success_count} 条")
    print(f"  失败: {fail_count} 条")

    if success_count > 0:
        stats = db.get_statistics()
        print(f"\n=== 数据库统计 ===")
        print(f"总合金数量: {stats['total_alloys']}")

        print(f"\n合金体系分布:")
        for system, count in stats.get('by_system', {}).items():
            if system:
                print(f"  {system}: {count} 条")

        if stats.get('zinc_range', {}).get('min'):
            print(f"\nZn含量范围: {stats['zinc_range']['min']:.1f}% - {stats['zinc_range']['max']:.1f}%")

        if stats.get('magnesium_range', {}).get('min'):
            print(f"Mg含量范围: {stats['magnesium_range']['min']:.1f}% - {stats['magnesium_range']['max']:.1f}%")

        if stats.get('uts_range', {}).get('min'):
            print(f"\n抗拉强度范围: {stats['uts_range']['min']:.0f} - {stats['uts_range']['max']:.0f} MPa")

        if stats.get('ys_range', {}).get('min'):
            print(f"屈服强度范围: {stats['ys_range']['min']:.0f} - {stats['ys_range']['max']:.0f} MPa")

        print(f"\n=== 导出功能 ===")
        csv_success = db.export_to_csv("alloy_training_data.csv")
        json_success = db.export_to_json("alloy_full_export.json")

        if csv_success:
            print("已导出CSV格式（适合ML训练）")
        if json_success:
            print("已导出JSON格式（完整数据）")

    db.close()

if __name__ == "__main__":
    main()
