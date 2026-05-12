import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple

def load_alloy_data(file_path: str) -> List[Dict]:
    """加载合金数据"""
    if not os.path.exists(file_path):
        print(f"错误：文件不存在: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_alloy_systems(data: List[Dict]) -> Counter:
    """分析合金体系分布"""
    systems = []
    for item in data:
        system = item.get('alloy_system', '未知')
        if system:
            systems.append(system)
    return Counter(systems)

def analyze_processing_methods(data: List[Dict]) -> Tuple[Counter, Counter]:
    """分析加工方式分布（包括单一工艺和组合工艺）"""
    single_methods = []
    combined_methods = []
    
    # 工艺名称标准化映射
    method_mapping = {
        'T4': '固溶处理',
        'T6': '固溶+时效',
        'T5': '时效处理',
        'solution treatment': '固溶处理',
        'aging treatment': '时效处理',
        'annealing': '退火',
        'quenching': '淬火',
        'hot rolling': '热轧',
        'cold rolling': '冷轧',
        'extrusion': '挤压',
        'casting': '铸造',
        'die casting': '压铸',
        'sand casting': '砂型铸造',
        'laser melting': '激光熔化',
        'SLM': '选择性激光熔化',
        'laser printing': '激光打印',
        '3D printing': '3D打印',
        'HIP': '热等静压',
        'hot isostatic pressing': '热等静压',
        'forging': '锻造',
        'rolling': '轧制',
        'welding': '焊接',
        'machining': '机械加工',
    }
    
    for item in data:
        processing = item.get('processing', {})
        if processing is None:
            continue
        
        methods_in_item = []
        
        # 提取各种加工方式
        if processing.get('casting_method'):
            method = processing['casting_method']
            normalized = method_mapping.get(method.lower(), method)
            methods_in_item.append(normalized)
        
        if processing.get('heat_treatment'):
            method = processing['heat_treatment']
            normalized = method_mapping.get(method.lower(), method)
            methods_in_item.append(normalized)
        
        if processing.get('aging_temperature'):
            methods_in_item.append('时效处理')
        
        if processing.get('annealing_temperature'):
            methods_in_item.append('退火')
        
        if processing.get('extrusion_temperature'):
            methods_in_item.append('挤压')
        
        # 处理自由文本描述的工艺
        processing_route = processing.get('processing_route', '')
        if processing_route:
            for keyword in ['SLM', 'HIP', 'laser', 'casting', 'extrusion', 'rolling', 'forging']:
                if keyword.lower() in processing_route.lower():
                    normalized = method_mapping.get(keyword.lower(), keyword)
                    if normalized not in methods_in_item:
                        methods_in_item.append(normalized)
        
        # 添加单一工艺和组合工艺
        for method in methods_in_item:
            single_methods.append(method)
        
        if len(methods_in_item) >= 2:
            combined = '+'.join(sorted(methods_in_item))
            combined_methods.append(combined)
    
    return Counter(single_methods), Counter(combined_methods)

def analyze_processing_mechanical_correlation(data: List[Dict]) -> Dict[str, List[Tuple[float, float, float]]]:
    """分析加工方式与力学性能的关联"""
    correlation = {}
    
    method_mapping = get_method_mapping()
    
    for item in data:
        processing = item.get('processing', {})
        if processing is None:
            continue
        
        mechanical = item.get('mechanical_properties', {})
        if mechanical is None:
            continue
        
        # 获取力学性能
        uts, ys, elongation = extract_mechanical_properties(mechanical)
        
        if uts is None:
            continue
        
        # 获取加工方式
        methods = extract_processing_methods(processing, method_mapping)
        
        # 添加到关联数据
        for method in methods:
            if method not in correlation:
                correlation[method] = []
            correlation[method].append((uts, ys, elongation))
    
    return correlation

def get_method_mapping() -> Dict[str, str]:
    """获取工艺名称标准化映射"""
    return {
        'T4': '固溶处理',
        'T6': '固溶+时效',
        'T5': '时效处理',
        'solution treatment': '固溶处理',
        'aging treatment': '时效处理',
        'annealing': '退火',
        'extrusion': '挤压',
        'casting': '铸造',
        'SLM': '选择性激光熔化',
        'laser printing': '激光打印',
        'HIP': '热等静压',
        'as-cast': '铸态',
    }

def extract_mechanical_properties(mechanical: Dict) -> Tuple[float, float, float]:
    """提取力学性能数据"""
    uts = None
    ys = None
    elongation = None
    
    if isinstance(mechanical, dict):
        uts = mechanical.get('ultimate_tensile_strength')
        ys = mechanical.get('yield_strength')
        elongation = mechanical.get('elongation')
    
    # 尝试转换为数值
    try:
        uts = float(uts) if uts else None
        ys = float(ys) if ys else None
        elongation = float(elongation) if elongation else None
    except (ValueError, TypeError):
        uts = ys = elongation = None
    
    return uts, ys, elongation

def extract_processing_methods(processing: Dict, method_mapping: Dict[str, str]) -> List[str]:
    """提取加工方式列表"""
    methods = []
    
    if processing.get('casting_method'):
        method = processing['casting_method']
        normalized = method_mapping.get(method.lower(), method)
        methods.append(normalized)
    
    if processing.get('heat_treatment'):
        method = processing['heat_treatment']
        normalized = method_mapping.get(method.lower(), method)
        methods.append(normalized)
    
    if processing.get('aging_temperature'):
        methods.append('时效处理')
    
    if processing.get('annealing_temperature'):
        methods.append('退火')
    
    return methods

def analyze_by_alloy_system(data: List[Dict], top_n_systems: int = 5) -> Dict[str, Dict]:
    """按合金体系分组分析（核心功能）"""
    result = {}
    method_mapping = get_method_mapping()
    
    # 按合金体系分组
    systems_data = {}
    for item in data:
        system = item.get('alloy_system', '未知')
        if system not in systems_data:
            systems_data[system] = []
        systems_data[system].append(item)
    
    # 取前N个最大的合金体系
    sorted_systems = sorted(systems_data.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_systems]
    
    for system, items in sorted_systems:
        system_result = {
            'total_count': len(items),
            'processing_methods': Counter(),
            'mechanical_properties': {
                'uts': [],
                'ys': [],
                'elongation': []
            },
            'processing_mechanical': {}  # 加工方式 -> 性能列表
        }
        
        for item in items:
            processing = item.get('processing', {})
            if processing is None:
                continue
            
            mechanical = item.get('mechanical_properties', {})
            uts, ys, elongation = extract_mechanical_properties(mechanical)
            
            # 统计加工方式
            methods = extract_processing_methods(processing, method_mapping)
            for method in methods:
                system_result['processing_methods'][method] += 1
            
            # 统计力学性能
            if uts:
                system_result['mechanical_properties']['uts'].append(uts)
            if ys:
                system_result['mechanical_properties']['ys'].append(ys)
            if elongation:
                system_result['mechanical_properties']['elongation'].append(elongation)
            
            # 加工方式与力学性能关联
            for method in methods:
                if method not in system_result['processing_mechanical']:
                    system_result['processing_mechanical'][method] = {
                        'uts': [],
                        'ys': [],
                        'elongation': []
                    }
                if uts:
                    system_result['processing_mechanical'][method]['uts'].append(uts)
                if ys:
                    system_result['processing_mechanical'][method]['ys'].append(ys)
                if elongation:
                    system_result['processing_mechanical'][method]['elongation'].append(elongation)
        
        result[system] = system_result
    
    return result

def plot_system_processing_bar(alloy_system_data: Dict, system_name: str):
    """绘制单个合金体系的加工方式分布柱状图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    processing_methods = alloy_system_data['processing_methods']
    if not processing_methods:
        return
    
    labels = list(processing_methods.keys())
    counts = list(processing_methods.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='#2ca02c')
    
    plt.title(f'{system_name} - 加工方式分布', fontsize=14)
    plt.xlabel('加工方式', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_system_mechanical_box(alloy_system_data: Dict, system_name: str):
    """绘制单个合金体系的力学性能箱线图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    props = alloy_system_data['mechanical_properties']
    data_to_plot = []
    labels = []
    
    if props['uts']:
        data_to_plot.append(props['uts'])
        labels.append('抗拉强度 (MPa)')
    if props['ys']:
        data_to_plot.append(props['ys'])
        labels.append('屈服强度 (MPa)')
    if props['elongation']:
        data_to_plot.append(props['elongation'])
        labels.append('延伸率 (%)')
    
    if not data_to_plot:
        return
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    
    plt.title(f'{system_name} - 力学性能分布', fontsize=14)
    plt.ylabel('数值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_system_processing_mechanical(alloy_system_data: Dict, system_name: str):
    """绘制单个合金体系的加工方式与力学性能关联散点图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    processing_mechanical = alloy_system_data['processing_mechanical']
    if not processing_mechanical:
        return
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plt.figure(figsize=(10, 6))
    
    plotted_count = 0
    
    for i, (method, props) in enumerate(processing_mechanical.items()):
        if len(props['uts']) < 2:
            continue
        
        uts_values = props['uts']
        elongation_values = props.get('elongation', [])
        
        if not elongation_values:
            elongation_values = [None] * len(uts_values)
        
        min_len = min(len(uts_values), len(elongation_values))
        uts_values = uts_values[:min_len]
        elongation_values = elongation_values[:min_len]
        
        plt.scatter(uts_values, elongation_values,
                    label=method, color=colors[i % len(colors)],
                    alpha=0.7, s=50)
        plotted_count += 1
    
    plt.title(f'{system_name} - 加工方式与力学性能关系', fontsize=14)
    plt.xlabel('抗拉强度 (MPa)', fontsize=12)
    plt.ylabel('延伸率 (%)', fontsize=12)
    
    if plotted_count > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_system_analysis(system_name: str, system_data: Dict):
    """打印单个合金体系的详细分析报告"""
    print(f"\n{'='*60}")
    print(f"合金体系: {system_name}")
    print(f"{'='*60}")
    
    # 基本信息
    print(f"\n一、基本信息")
    print(f"  记录数量: {system_data['total_count']} 条")
    
    # 加工方式分布
    print(f"\n二、加工方式分布")
    if system_data['processing_methods']:
        total = sum(system_data['processing_methods'].values())
        for method, count in system_data['processing_methods'].most_common(5):
            percentage = (count / total) * 100
            print(f"  {method}: {count} 次 ({percentage:.1f}%)")
    else:
        print(f"  暂无加工方式数据")
    
    # 力学性能统计
    print(f"\n三、力学性能统计")
    props = system_data['mechanical_properties']
    
    if props['uts']:
        print(f"  抗拉强度 (UTS):")
        print(f"    样本数: {len(props['uts'])}")
        print(f"    平均值: {sum(props['uts'])/len(props['uts']):.1f} MPa")
        print(f"    范围: {min(props['uts']):.1f} - {max(props['uts']):.1f} MPa")
    
    if props['ys']:
        print(f"  屈服强度 (YS):")
        print(f"    样本数: {len(props['ys'])}")
        print(f"    平均值: {sum(props['ys'])/len(props['ys']):.1f} MPa")
        print(f"    范围: {min(props['ys']):.1f} - {max(props['ys']):.1f} MPa")
    
    if props['elongation']:
        print(f"  延伸率:")
        print(f"    样本数: {len(props['elongation'])}")
        print(f"    平均值: {sum(props['elongation'])/len(props['elongation']):.1f}%")
        print(f"    范围: {min(props['elongation']):.1f} - {max(props['elongation']):.1f}%")
    
    # 加工方式与力学性能关联
    print(f"\n四、加工方式与力学性能关联")
    processing_mechanical = system_data['processing_mechanical']
    if processing_mechanical:
        for method, props in processing_mechanical.items():
            if props['uts']:
                avg_uts = sum(props['uts']) / len(props['uts'])
                print(f"  {method}:")
                print(f"    样本数: {len(props['uts'])}")
                print(f"    平均抗拉强度: {avg_uts:.1f} MPa")
                if props['elongation']:
                    avg_elong = sum(props['elongation']) / len(props['elongation'])
                    print(f"    平均延伸率: {avg_elong:.1f}%")
    else:
        print(f"  暂无加工方式与力学性能关联数据")

def plot_comprehensive_heatmap(data: List[Dict], top_n_systems: int = 8, top_n_methods: int = 8):
    """绘制综合热力图：合金体系 × 加工方式 × 力学性能"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    method_mapping = get_method_mapping()
    
    # 按合金体系分组
    systems_data = {}
    for item in data:
        system = item.get('alloy_system', '未知')
        if system not in systems_data:
            systems_data[system] = []
        systems_data[system].append(item)
    
    # 取前N个最大的合金体系
    sorted_systems = sorted(systems_data.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_systems]
    
    # 统计所有加工方式
    all_methods = Counter()
    for system, items in sorted_systems:
        for item in items:
            processing = item.get('processing', {})
            if processing:
                methods = extract_processing_methods(processing, method_mapping)
                for method in methods:
                    all_methods[method] += 1
    
    # 取前N个最常用的加工方式
    top_methods = [method for method, _ in all_methods.most_common(top_n_methods)]
    
    # 构建热力图数据矩阵
    system_names = [system for system, _ in sorted_systems]
    heatmap_data = np.zeros((len(system_names), len(top_methods)))
    
    for i, (system, items) in enumerate(sorted_systems):
        for item in items:
            processing = item.get('processing', {})
            if not processing:
                continue
            
            mechanical = item.get('mechanical_properties', {})
            uts, ys, elongation = extract_mechanical_properties(mechanical)
            
            if uts is None:
                continue
            
            methods = extract_processing_methods(processing, method_mapping)
            for method in methods:
                if method in top_methods:
                    j = top_methods.index(method)
                    heatmap_data[i, j] = max(heatmap_data[i, j], uts)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 使用 seaborn 绘制热力图
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=top_methods,
        yticklabels=system_names,
        cbar_kws={'label': '平均抗拉强度 (MPa)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('合金体系 × 加工方式 × 力学性能热力图', fontsize=16, pad=20)
    plt.xlabel('加工方式', fontsize=12)
    plt.ylabel('合金体系', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_comprehensive_heatmap_elongation(data: List[Dict], top_n_systems: int = 8, top_n_methods: int = 8):
    """绘制综合热力图：合金体系 × 加工方式 × 延伸率"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    method_mapping = get_method_mapping()
    
    # 按合金体系分组
    systems_data = {}
    for item in data:
        system = item.get('alloy_system', '未知')
        if system not in systems_data:
            systems_data[system] = []
        systems_data[system].append(item)
    
    # 取前N个最大的合金体系
    sorted_systems = sorted(systems_data.items(), key=lambda x: len(x[1]), reverse=True)[:top_n_systems]
    
    # 统计所有加工方式
    all_methods = Counter()
    for system, items in sorted_systems:
        for item in items:
            processing = item.get('processing', {})
            if processing:
                methods = extract_processing_methods(processing, method_mapping)
                for method in methods:
                    all_methods[method] += 1
    
    # 取前N个最常用的加工方式
    top_methods = [method for method, _ in all_methods.most_common(top_n_methods)]
    
    # 构建热力图数据矩阵
    system_names = [system for system, _ in sorted_systems]
    heatmap_data = np.zeros((len(system_names), len(top_methods)))
    
    for i, (system, items) in enumerate(sorted_systems):
        for item in items:
            processing = item.get('processing', {})
            if not processing:
                continue
            
            mechanical = item.get('mechanical_properties', {})
            uts, ys, elongation = extract_mechanical_properties(mechanical)
            
            if elongation is None:
                continue
            
            methods = extract_processing_methods(processing, method_mapping)
            for method in methods:
                if method in top_methods:
                    j = top_methods.index(method)
                    heatmap_data[i, j] = max(heatmap_data[i, j], elongation)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 使用 seaborn 绘制热力图
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        xticklabels=top_methods,
        yticklabels=system_names,
        cbar_kws={'label': '延伸率 (%)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('合金体系 × 加工方式 × 延伸率热力图', fontsize=16, pad=20)
    plt.xlabel('加工方式', fontsize=12)
    plt.ylabel('合金体系', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_bar_chart(data: Counter, title: str, xlabel: str, ylabel: str, top_n: int = 10, color: str = '#1f77b4'):
    """绘制柱状图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    top_items = data.most_common(top_n)
    if not top_items:
        print(f"警告：{title} 没有数据")
        return
    
    labels = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color=color)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom')
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_pie_chart(data: Counter, title: str, top_n: int = 10):
    """绘制饼图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    top_items = data.most_common(top_n)
    if not top_items:
        print(f"警告：{title} 没有数据")
        return
    
    labels = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_processing_mechanical_scatter(correlation: Dict[str, List[Tuple[float, float, float]]]):
    """绘制加工方式与力学性能的散点图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(10, 6))
    
    for i, (method, properties) in enumerate(correlation.items()):
        if len(properties) < 2:
            continue
        
        uts_values = [p[0] for p in properties if p[0]]
        elongation_values = [p[2] for p in properties if p[2]]
        
        if len(uts_values) != len(elongation_values):
            min_len = min(len(uts_values), len(elongation_values))
            uts_values = uts_values[:min_len]
            elongation_values = elongation_values[:min_len]
        
        plt.scatter(uts_values, elongation_values, 
                    label=method, color=colors[i % len(colors)], 
                    alpha=0.7, s=50)
    
    plt.title('加工方式与力学性能关系（UTS vs 延伸率）', fontsize=14)
    plt.xlabel('抗拉强度 UTS (MPa)', fontsize=12)
    plt.ylabel('延伸率 (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_statistics(alloy_counter: Counter, single_methods: Counter, 
                     combined_methods: Counter, correlation: Dict):
    """打印统计信息"""
    print("=" * 60)
    print("合金数据分析报告")
    print("=" * 60)
    
    # 合金体系统计
    print("\n一、合金体系分布")
    print("-" * 40)
    total_alloys = sum(alloy_counter.values())
    for system, count in alloy_counter.most_common(10):
        percentage = (count / total_alloys) * 100
        print(f"  {system}: {count} 条 ({percentage:.1f}%)")
    
    # 单一加工方式统计
    print("\n二、单一加工方式分布")
    print("-" * 40)
    total_single = sum(single_methods.values())
    if total_single > 0:
        for method, count in single_methods.most_common(10):
            percentage = (count / total_single) * 100
            print(f"  {method}: {count} 次 ({percentage:.1f}%)")
    else:
        print("  暂无单一加工方式数据")
    
    # 组合加工方式统计
    print("\n三、组合加工方式分布")
    print("-" * 40)
    total_combined = sum(combined_methods.values())
    if total_combined > 0:
        for method, count in combined_methods.most_common(10):
            percentage = (count / total_combined) * 100
            print(f"  {method}: {count} 次 ({percentage:.1f}%)")
    else:
        print("  暂无组合加工方式数据")
    
    # 加工方式与力学性能关联统计
    print("\n四、加工方式与力学性能关联")
    print("-" * 40)
    if correlation:
        for method, properties in correlation.items():
            if len(properties) >= 2:
                uts_list = [p[0] for p in properties if p[0]]
                ys_list = [p[1] for p in properties if p[1]]
                elong_list = [p[2] for p in properties if p[2]]
                
                if uts_list:
                    avg_uts = sum(uts_list) / len(uts_list)
                    print(f"  {method}:")
                    print(f"    样本数: {len(properties)}")
                    print(f"    平均抗拉强度: {avg_uts:.1f} MPa")
                    if ys_list:
                        avg_ys = sum(ys_list) / len(ys_list)
                        print(f"    平均屈服强度: {avg_ys:.1f} MPa")
                    if elong_list:
                        avg_elong = sum(elong_list) / len(elong_list)
                        print(f"    平均延伸率: {avg_elong:.1f}%")
                    print()
    else:
        print("  暂无加工方式与力学性能关联数据")
    
    # 总体统计
    print("\n五、总体统计")
    print("-" * 40)
    print(f"  数据总量: {len(alloy_counter)} 种合金体系")
    print(f"  记录总数: {total_alloys} 条")
    print(f"  单一加工方式种类: {len(single_methods)} 种")
    print(f"  组合加工方式种类: {len(combined_methods)} 种")
    print("=" * 60)

def main():
    """主函数"""
    data_file = "batch_alloy_data.json"
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        alt_paths = [
            "../batch_alloy_data.json",
            "backend/batch_alloy_data.json",
            "../backend/batch_alloy_data.json"
        ]
        for path in alt_paths:
            if os.path.exists(path):
                data_file = path
                break
    
    print(f"正在加载数据: {data_file}")
    data = load_alloy_data(data_file)
    
    if not data:
        print("未找到数据，请检查文件路径")
        return
    
    # 分析数据
    alloy_counter = analyze_alloy_systems(data)
    single_methods, combined_methods = analyze_processing_methods(data)
    correlation = analyze_processing_mechanical_correlation(data)
    
    # 打印统计信息
    print_statistics(alloy_counter, single_methods, combined_methods, correlation)
    
    # 生成图表
    try:
        # 合金体系柱状图
        plot_bar_chart(alloy_counter, '合金体系分布', '合金体系', '数量')
        
        # 单一加工方式饼图
        plot_pie_chart(single_methods, '单一加工方式分布')
        
        # 组合加工方式柱状图
        plot_bar_chart(combined_methods, '组合加工方式分布', '组合工艺', '数量', color='#ff7f0e')
        
        # 加工方式与力学性能散点图
        plot_processing_mechanical_scatter(correlation)
        
    except ImportError as e:
        print(f"\n警告：matplotlib 未安装，无法生成图表")
        print("请安装：pip install matplotlib")
    except Exception as e:
        print(f"\n图表生成出错: {e}")
    
    # ========== 综合热力图分析 ==========
    print("\n" + "="*60)
    print("综合热力图分析")
    print("="*60)
    
    try:
        # 抗拉强度热力图
        print("\n正在生成合金体系 × 加工方式 × 抗拉强度热力图...")
        plot_comprehensive_heatmap(data, top_n_systems=8, top_n_methods=8)
        
        # 延伸率热力图
        print("\n正在生成合金体系 × 加工方式 × 延伸率热力图...")
        plot_comprehensive_heatmap_elongation(data, top_n_systems=8, top_n_methods=8)
        
    except Exception as e:
        print(f"\n热力图生成出错: {e}")
    
    # ========== 按合金体系分组分析 ==========
    print("\n" + "="*60)
    print("按合金体系分组分析")
    print("="*60)
    
    # 分析前5大合金体系
    systems_data = analyze_by_alloy_system(data, top_n_systems=5)
    
    # 打印每个体系的详细分析报告
    for system_name, system_data in systems_data.items():
        print_system_analysis(system_name, system_data)
    
    # 为每个体系生成图表
    try:
        for system_name, system_data in systems_data.items():
            # 加工方式分布
            plot_system_processing_bar(system_data, system_name)
            
            # 力学性能箱线图
            plot_system_mechanical_box(system_data, system_name)
            
            # 加工方式与力学性能关联
            plot_system_processing_mechanical(system_data, system_name)
            
    except Exception as e:
        print(f"\n图表生成出错: {e}")

if __name__ == "__main__":
    main()
